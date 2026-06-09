"""
实验 6: 误差传播定量分析
========================
固定 d=9, domain=[-8,0]
逐层记录 hidden states 的相对误差，拟合误差增长率曲线
目标: 证明多项式误差不随层数累积（或增长率很低）
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

def make_poly_softmax(M_domain, d=9):
    lam_val = 1.0; coeffs = generate_polynomial(lam_val, M_domain, d)
    cn = np.array(coeffs, dtype=np.float64)
    def fn(logits, dim=-1, dtype=None, **kw):
        x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
        xs = x - mx; xc = torch.clamp(xs, -M_domain, 0.0)
        y = horner_torch(cn, -xc); y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return fn

# ============================================================================
# Per-layer error analysis
# ============================================================================

@torch.no_grad()
def analyze_error_propagation(model, tokenizer, prompts, poly_softmax_fn, device='cuda'):
    """Run model on multiple prompts, capture per-layer hidden states, compute relative error."""
    orig_softmax = F.softmax

    # Storage: [prompt_idx][layer_idx] -> dict
    all_prompt_errors = []

    for p_idx, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
        input_ids = enc['input_ids'].to(device)
        seq_len = input_ids.size(1)

        # Original forward
        out_orig = model(input_ids, output_hidden_states=True)
        hidden_orig = [h.detach().float() for h in out_orig.hidden_states if h is not None]

        # PolyAttn forward
        F.softmax = poly_softmax_fn
        out_poly = model(input_ids, output_hidden_states=True)
        hidden_poly = [h.detach().float() for h in out_poly.hidden_states if h is not None]
        F.softmax = orig_softmax

        # Logits error
        logits_diff = (out_orig.logits.float() - out_poly.logits.float())
        logits_rel = float(logits_diff.norm() / (out_orig.logits.float().norm() + 1e-10))
        logits_cos = float(F.cosine_similarity(
            out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))

        # Per-layer hidden state error
        layer_errors = []
        for l_idx, (h_orig, h_poly) in enumerate(zip(hidden_orig, hidden_poly)):
            diff = h_orig - h_poly
            abs_error = float(diff.norm())
            rel_error = float(diff.norm() / (h_orig.norm() + 1e-10))
            max_abs = float(diff.abs().max())
            cos_sim = float(F.cosine_similarity(
                h_orig.reshape(-1), h_poly.reshape(-1), dim=0))

            layer_errors.append({
                'layer': l_idx,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'max_abs_error': max_abs,
                'cos_sim': cos_sim,
                'hidden_norm': float(h_orig.norm()),
            })

        all_prompt_errors.append({
            'prompt_idx': p_idx,
            'prompt': prompt[:80],
            'seq_len': seq_len,
            'logits_rel_error': logits_rel,
            'logits_cos_sim': logits_cos,
            'layer_errors': layer_errors,
        })

    return all_prompt_errors

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
    print("实验 6: 误差传播定量分析")
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
    num_layers = len(model.model.layers)
    print(f"  Layers: {num_layers}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Generate polynomial
    poly_fn = make_poly_softmax(DOMAIN_M, d=DEGREE)

    # Test prompts of varying lengths
    prompts = [
        # Short (10-15 tokens)
        "The future of artificial intelligence lies in the development of more efficient",
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",

        # Medium (20-30 tokens)
        "Machine learning has revolutionized how we approach complex problems in science and engineering. "
        "Deep neural networks have achieved remarkable results across many domains.",

        "深度学习的发展推动了自然语言处理领域的巨大进步，从机器翻译到文本生成都取得了突破性成果。"
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",

        # Longer (40-60 tokens)
        "The transformer architecture has become the foundation of modern NLP systems. "
        "Attention mechanisms allow models to focus on relevant parts of the input sequence, "
        "enabling better handling of long range dependencies and improving performance on various tasks.",

        "计算机视觉技术使得机器能够理解和分析图像内容，广泛应用于自动驾驶、医疗诊断等领域。"
        "量子计算作为一种新型计算范式，有望在密码学、药物研发等领域带来革命性变化。"
        "中国在人工智能领域的研究投入不断增加，推动了技术创新和产业升级。",
    ]

    # Run analysis
    all_prompt_errors = analyze_error_propagation(model, tokenizer, prompts, poly_fn, DEVICE)

    # Aggregate across prompts (mean and std per layer)
    print("\n" + "=" * 70)
    print("逐层误差传播 (跨 prompt 平均)")
    print("=" * 70)
    print(f"{'Layer':<8} {'RelErr':<12} {'CosSim':<12} {'MaxAbs':<12} {'AbsErr':<12} {'Norm':<12}")
    print("-" * 68)

    layer_aggregates = []
    for l in range(num_layers + 1):  # +1 for embedding layer
        rel_vals = [r['layer_errors'][l]['rel_error'] for r in all_prompt_errors if l < len(r['layer_errors'])]
        cos_vals = [r['layer_errors'][l]['cos_sim'] for r in all_prompt_errors if l < len(r['layer_errors'])]
        max_vals = [r['layer_errors'][l]['max_abs_error'] for r in all_prompt_errors if l < len(r['layer_errors'])]
        abs_vals = [r['layer_errors'][l]['abs_error'] for r in all_prompt_errors if l < len(r['layer_errors'])]
        norm_vals = [r['layer_errors'][l]['hidden_norm'] for r in all_prompt_errors if l < len(r['layer_errors'])]

        if not rel_vals: continue

        agg = {
            'layer': l,
            'rel_error_mean': float(np.mean(rel_vals)),
            'rel_error_std': float(np.std(rel_vals)),
            'cos_sim_mean': float(np.mean(cos_vals)),
            'cos_sim_std': float(np.std(cos_vals)),
            'max_abs_mean': float(np.mean(max_vals)),
            'abs_error_mean': float(np.mean(abs_vals)),
            'hidden_norm_mean': float(np.mean(norm_vals)),
        }
        layer_aggregates.append(agg)

        if l % 4 == 0 or l == num_layers:
            print(f"{'E' if l == 0 else str(l-1):<8} {agg['rel_error_mean']:<12.6e} "
                  f"{agg['cos_sim_mean']:<12.6f} {agg['max_abs_mean']:<12.6e} "
                  f"{agg['abs_error_mean']:<12.4f} {agg['hidden_norm_mean']:<12.2f}")

    # Error growth analysis
    print("\n--- 误差增长分析 ---")
    # Skip embedding (layer 0), focus on transformer layers
    transformer_layers = [a for a in layer_aggregates if a['layer'] > 0]
    rel_errors = [a['rel_error_mean'] for a in transformer_layers]

    # Fit exponential growth: err(l) = err_0 * exp(alpha * l)
    layers = np.arange(len(rel_errors))
    log_err = np.log(np.maximum(rel_errors, 1e-30))
    if len(layers) > 2:
        # Linear fit on log scale
        coeffs = np.polyfit(layers, log_err, 1)
        alpha = coeffs[0]  # growth rate
        growth_per_layer = np.exp(alpha) - 1
        print(f"  对数线性拟合: log(err) = {coeffs[0]:.6f} * layer + {coeffs[1]:.4f}")
        print(f"  每层误差增长率: {growth_per_layer*100:+.3f}%")
        if abs(growth_per_layer) < 0.01:
            print(f"  ✓ 误差增长率 < 1%/层 → 误差不随层数累积")
        else:
            print(f"  ✗ 误差增长率显著 → 深层可能有累积效应")

    # Compare first half vs second half
    mid = len(transformer_layers) // 2
    first_half = np.mean(rel_errors[:mid])
    second_half = np.mean(rel_errors[mid:])
    ratio = second_half / (first_half + 1e-30)
    print(f"  前半段平均相对误差: {first_half:.6e}")
    print(f"  后半段平均相对误差: {second_half:.6e}")
    print(f"  后半/前半比例: {ratio:.3f}")
    if ratio < 2.0:
        print(f"  ✓ 误差增长 < 2× → 无显著累积")

    # Regression for the claim "error does not accumulate"
    # Fit linear: err = a*l + b
    linear_coeffs = np.polyfit(layers, rel_errors, 1)
    linear_slope = linear_coeffs[0]
    print(f"\n  线性拟合: err = {linear_slope:.6e} * layer + {linear_coeffs[1]:.6e}")
    if abs(linear_slope) < 1e-4:
        print(f"  ✓ 线性斜率 < 1e-4 → 误差几乎恒定")

    # Output logits summary
    logits_rel = [r['logits_rel_error'] for r in all_prompt_errors]
    logits_cos = [r['logits_cos_sim'] for r in all_prompt_errors]
    print(f"\nOutput logits:")
    print(f"  相对误差: {np.mean(logits_rel):.6e} ± {np.std(logits_rel):.6e}")
    print(f"  Cos sim:  {np.mean(logits_cos):.6f} ± {np.std(logits_cos):.6f}")

    # Save
    output = {
        'experiment': 'error_propagation',
        'model': MODEL,
        'degree': DEGREE,
        'domain': DOMAIN_M,
        'num_layers': num_layers,
        'num_prompts': len(prompts),
        'per_prompt_results': all_prompt_errors,
        'layer_aggregates': layer_aggregates,
        'growth_analysis': {
            'log_linear_alpha': float(alpha) if len(layers) > 2 else None,
            'growth_per_layer_pct': float(growth_per_layer * 100) if len(layers) > 2 else None,
            'linear_slope': float(linear_slope),
            'first_half_mean': float(first_half),
            'second_half_mean': float(second_half),
            'ratio_second_to_first': float(ratio),
        },
        'output_logits': {
            'rel_error_mean': float(np.mean(logits_rel)),
            'rel_error_std': float(np.std(logits_rel)),
            'cos_sim_mean': float(np.mean(logits_cos)),
            'cos_sim_std': float(np.std(logits_cos)),
        },
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp6_error_propagation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    print("实验 6 完成！")

if __name__ == '__main__':
    main()
