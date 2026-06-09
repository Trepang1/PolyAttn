"""
实验 5: Attention 分布对比分析
==============================
固定 d=9, domain=[-8,0]
对比原始 exp Softmax 与 PolyAttn 多项式 Softmax 的 attention 分布
指标: 逐层 KL 散度、JS 距离、attention map 相关系数
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Polynomial utilities
# ============================================================================

def generate_polynomial(lambda_val, M, d):
    n_cheb = max(d * 12, 64)
    k = np.arange(n_cheb)
    t_nodes = np.cos(np.pi * (k + 0.5) / n_cheb)
    x_nodes = (M / 2.0) * (t_nodes + 1.0)
    f_vals = np.exp(-lambda_val * x_nodes)
    cheb = np.zeros(n_cheb)
    for k_idx in range(n_cheb):
        cheb[k_idx] = (2.0 / n_cheb) * np.sum(f_vals * np.cos(k_idx * np.pi * (k + 0.5) / n_cheb))
    cheb[0] /= 2.0; cheb = cheb[:d + 1]
    T = np.zeros((d + 1, d + 1)); T[0, 0] = 1.0
    if d >= 1: T[1, 1] = 1.0
    for k_idx in range(2, d + 1):
        T[k_idx, 1:] += 2.0 * T[k_idx - 1, :-1]; T[k_idx, :] -= T[k_idx - 2, :]
    mono = np.zeros(d + 1)
    for k_idx in range(d + 1):
        for j in range(k_idx + 1):
            t_kj = T[k_idx, j]
            if abs(t_kj) < 1e-16: continue
            for r in range(j + 1):
                mono[r] += t_kj * cheb[k_idx] * comb(j, r) * (2.0 / M) ** r * (-1.0) ** (j - r)
    return list(mono)

def horner_torch(c, x):
    y = torch.full_like(x, c[-1])
    for ci in reversed(c[:-1]): y = y * x + ci
    return y

def make_poly_softmax_simple(M_domain, d=9):
    """Simplified: returns a module-style function that replaces the exp part only."""
    lam_val = 1.0; coeffs = generate_polynomial(lam_val, M_domain, d)
    cn = np.array(coeffs, dtype=np.float64)
    def poly_fn(logits, dim=-1, dtype=None, **kw):
        x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
        xs = x - mx; xc = torch.clamp(xs, -M_domain, 0.0)
        y = horner_torch(cn, -xc); y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return poly_fn

# ============================================================================
# Attention capture via hooks
# ============================================================================

def capture_attention_maps(model, input_ids, poly_softmax_fn=None):
    """Capture attention maps from all layers. Returns list of attention weight tensors."""
    attention_maps = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # For eager attention, output is a tuple (attn_output, attn_weights, ...)
            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    attention_maps.append({
                        'layer': layer_idx,
                        'weights': attn_weights.detach().float().cpu(),
                    })
        return hook

    # Register hooks on all attention layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        h = attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run inference
    if poly_softmax_fn is not None:
        orig_softmax = F.softmax
        F.softmax = poly_softmax_fn
        try:
            with torch.no_grad():
                _ = model(input_ids)
        finally:
            F.softmax = orig_softmax
    else:
        with torch.no_grad():
            _ = model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    return attention_maps

# ============================================================================
# Distribution metrics
# ============================================================================

def compute_kl_divergence(p, q, eps=1e-12):
    """KL(P||Q) per head. p, q: (num_heads, seq_len, seq_len)"""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    # KL(P||Q) = sum P * log(P/Q), last dim
    kl = (p * (p.log() - q.log())).sum(dim=-1)  # (num_heads, seq_len)
    return kl.mean(dim=-1)  # (num_heads,) 平均 over query positions

def compute_js_distance(p, q, eps=1e-12):
    """Jensen-Shannon distance per head. Range [0, 1]."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)  # per query position
    return torch.sqrt(js.mean(dim=-1))  # sqrt for metric

def compute_correlation(p, q):
    """Pearson correlation per head."""
    # Flatten seq_len × seq_len
    p_flat = p.reshape(p.shape[0], -1)  # (num_heads, seq_len*seq_len)
    q_flat = q.reshape(q.shape[0], -1)
    p_centered = p_flat - p_flat.mean(dim=-1, keepdim=True)
    q_centered = q_flat - q_flat.mean(dim=-1, keepdim=True)
    cov = (p_centered * q_centered).sum(dim=-1)
    std_p = torch.sqrt((p_centered ** 2).sum(dim=-1))
    std_q = torch.sqrt((q_centered ** 2).sum(dim=-1))
    return cov / (std_p * std_q + 1e-12)

# ============================================================================
# Main
# ============================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DOMAIN_M = 8.0
    DEGREE = 9
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("实验 5: Attention 分布对比分析")
    print(f"固定: d={DEGREE}, domain=[-{DOMAIN_M},0]")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Generate polynomial
    poly_fn = make_poly_softmax_simple(DOMAIN_M, d=DEGREE)

    # Test prompts
    prompts = [
        "The future of artificial intelligence lies in the development of more efficient",
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",
        "Machine learning has revolutionized how we approach complex problems in",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",
        "Transformer architecture has become the foundation of modern NLP systems and",
    ]

    all_results = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*50}")
        print(f"Prompt {prompt_idx+1}/{len(prompts)}: {prompt[:60]}...")

        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        input_ids = enc['input_ids'].to(DEVICE)
        seq_len = input_ids.size(1)
        print(f"  Sequence length: {seq_len}")

        # Capture original attention
        orig_attn = capture_attention_maps(model, input_ids, poly_softmax_fn=None)

        # Capture PolyAttn attention
        poly_attn = capture_attention_maps(model, input_ids, poly_softmax_fn=poly_fn)

        # Compare per layer
        prompt_results = []
        for orig, poly in zip(orig_attn, poly_attn):
            assert orig['layer'] == poly['layer']
            layer_idx = orig['layer']
            w_orig = orig['weights']  # (1, num_heads, seq, seq)
            w_poly = poly['weights']  # (1, num_heads, seq, seq)

            # Remove batch dim
            w_orig = w_orig.squeeze(0)
            w_poly = w_poly.squeeze(0)

            # Compute metrics per head
            kl = compute_kl_divergence(w_orig, w_poly)  # (num_heads,)
            js = compute_js_distance(w_orig, w_poly)    # (num_heads,)
            corr = compute_correlation(w_orig, w_poly)  # (num_heads,)

            layer_data = {
                'layer': layer_idx,
                'num_heads': w_orig.shape[0],
                'kl_mean': float(kl.mean()),
                'kl_std': float(kl.std()),
                'kl_max': float(kl.max()),
                'js_mean': float(js.mean()),
                'js_std': float(js.std()),
                'js_max': float(js.max()),
                'corr_mean': float(corr.mean()),
                'corr_std': float(corr.std()),
                'corr_min': float(corr.min()),
            }
            prompt_results.append(layer_data)

            if layer_idx % 8 == 0 or layer_idx == len(orig_attn) - 1:
                print(f"  L{layer_idx:2d}: KL={layer_data['kl_mean']:.6f}, "
                      f"JS={layer_data['js_mean']:.6f}, Corr={layer_data['corr_mean']:.4f}")

        all_results.append({
            'prompt_idx': prompt_idx,
            'prompt': prompt,
            'seq_len': seq_len,
            'layers': prompt_results,
        })

    # Aggregate across prompts
    print("\n" + "=" * 70)
    print("跨 Prompt 汇总: Attention 分布一致性")
    print("=" * 70)

    num_layers = len(all_results[0]['layers'])
    layer_aggregates = []
    for l in range(num_layers):
        kl_vals = [r['layers'][l]['kl_mean'] for r in all_results]
        js_vals = [r['layers'][l]['js_mean'] for r in all_results]
        corr_vals = [r['layers'][l]['corr_mean'] for r in all_results]

        agg = {
            'layer': l,
            'kl_mean_across_prompts': float(np.mean(kl_vals)),
            'kl_std_across_prompts': float(np.std(kl_vals)),
            'js_mean_across_prompts': float(np.mean(js_vals)),
            'js_std_across_prompts': float(np.std(js_vals)),
            'corr_mean_across_prompts': float(np.mean(corr_vals)),
            'corr_std_across_prompts': float(np.std(corr_vals)),
        }
        layer_aggregates.append(agg)

        if l % 8 == 0 or l == num_layers - 1:
            print(f"  L{l:2d}: KL={agg['kl_mean_across_prompts']:.6f}±{agg['kl_std_across_prompts']:.6f}, "
                  f"JS={agg['js_mean_across_prompts']:.6f}±{agg['js_std_across_prompts']:.6f}, "
                  f"Corr={agg['corr_mean_across_prompts']:.4f}±{agg['corr_std_across_prompts']:.4f}")

    # Overall summary
    all_kl = [agg['kl_mean_across_prompts'] for agg in layer_aggregates]
    all_js = [agg['js_mean_across_prompts'] for agg in layer_aggregates]
    all_corr = [agg['corr_mean_across_prompts'] for agg in layer_aggregates]

    print(f"\n总体统计:")
    print(f"  KL divergence  (层平均): {np.mean(all_kl):.6f} (越小越好)")
    print(f"  JS distance    (层平均): {np.mean(all_js):.6f} (越小越好)")
    print(f"  Pearson corr   (层平均): {np.mean(all_corr):.4f} (越接近1越好)")

    # Check for problematic layers
    max_kl_layer = max(layer_aggregates, key=lambda x: x['kl_mean_across_prompts'])
    min_corr_layer = min(layer_aggregates, key=lambda x: x['corr_mean_across_prompts'])
    print(f"\n差异最大层: L{max_kl_layer['layer']} (KL={max_kl_layer['kl_mean_across_prompts']:.6f})")
    print(f"相关最低层: L{min_corr_layer['layer']} (Corr={min_corr_layer['corr_mean_across_prompts']:.4f})")

    # Save
    output = {
        'experiment': 'attention_distribution',
        'model': MODEL,
        'degree': DEGREE,
        'domain': DOMAIN_M,
        'num_prompts': len(prompts),
        'per_prompt_results': all_results,
        'layer_aggregates': layer_aggregates,
        'overall': {
            'kl_mean': float(np.mean(all_kl)),
            'js_mean': float(np.mean(all_js)),
            'corr_mean': float(np.mean(all_corr)),
        },
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp5_attention_distribution.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    print("实验 5 完成！")

if __name__ == '__main__':
    main()
