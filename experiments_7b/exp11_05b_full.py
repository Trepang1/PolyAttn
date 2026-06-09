"""
实验 11: Qwen2.5-0.5B 完整跨模型验证
=====================================
Degree 消融 + Attention 分布 + 误差传播 (精简版)
验证所有 7B 的关键发现是否在 0.5B 上复现
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

REAL_SOFTMAX = F.softmax

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

def check_approx(coeffs, lam, M):
    grid = np.linspace(0, M, 5001)
    return float(np.max(np.abs(np.exp(-lam * grid) - np.polyval(coeffs[::-1], grid))))

def horner_torch(c, x):
    y = torch.full_like(x, c[-1])
    for ci in reversed(c[:-1]): y = y * x + ci
    return y

def make_poly_softmax(domain_M, d=9):
    lam = 1.0; coeffs = generate_polynomial(lam, domain_M, d)
    linf = check_approx(coeffs, lam, domain_M)
    cn = np.array(coeffs, dtype=np.float64)
    def fn(logits, dim=-1, dtype=None, **kw):
        x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
        xs = x - mx; xc = torch.clamp(xs, -domain_M, 0.0)
        y = horner_torch(cn, -xc); y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return fn, linf

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, softmax_fn, max_length=128, device='cuda'):
    model.eval()
    F.softmax = softmax_fn
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        ids = enc['input_ids'].to(device)
        if ids.size(1) < 4: continue
        try:
            out = model(ids, labels=ids)
            if out.loss is not None and not torch.isnan(out.loss):
                total_loss += out.loss.item() * ids.size(1)
                total_tokens += ids.size(1)
        except: pass
    if total_tokens == 0: return float('inf')
    return float(np.exp(total_loss / total_tokens))

@torch.no_grad()
def compare_with_real(model, tokenizer, prompt, poly_fn, device='cuda'):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)
    F.softmax = REAL_SOFTMAX
    out_orig = model(input_ids)
    F.softmax = poly_fn
    out_poly = model(input_ids)
    F.softmax = REAL_SOFTMAX
    cos_sim = float(F.cosine_similarity(out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    _, orig_top = out_orig.logits[0, -1].topk(10)
    _, poly_top = out_poly.logits[0, -1].topk(10)
    return cos_sim, len(set(orig_top.tolist()) & set(poly_top.tolist()))

def captured_attention_kl(model, tokenizer, prompt, poly_fn, device='cuda'):
    """Quick per-layer attention KL comparison."""
    attention_orig = []
    attention_poly = []

    def make_hook(store):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                store.append(output[1].detach().float().cpu())
        return hook

    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
    input_ids = enc['input_ids'].to(device)

    hooks = [layer.self_attn.register_forward_hook(make_hook(attention_orig)) for layer in model.model.layers]

    F.softmax = REAL_SOFTMAX
    model(input_ids)
    for h in hooks: h.remove()

    hooks = [layer.self_attn.register_forward_hook(make_hook(attention_poly)) for layer in model.model.layers]
    F.softmax = poly_fn
    model(input_ids)
    for h in hooks: h.remove()
    F.softmax = REAL_SOFTMAX

    # KL per layer
    kl_layers = []
    for i, (ao, ap) in enumerate(zip(attention_orig, attention_poly)):
        ao = ao.squeeze(0).clamp(min=1e-12)
        ap = ap.squeeze(0).clamp(min=1e-12)
        kl = float((ao * (ao.log() - ap.log())).sum(dim=-1).mean())
        # Pearson
        ao_f = ao.reshape(ao.shape[0], -1)
        ap_f = ap.reshape(ap.shape[0], -1)
        corr = float(torch.nn.functional.cosine_similarity(ao_f, ap_f, dim=-1).mean())
        kl_layers.append({'layer': i, 'kl': kl, 'pearson': corr})
    return kl_layers

def main():
    MODEL = '/root/autodl-tmp/Qwen2.5-0.5B'
    DEVICE = 'cuda'
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("实验 11: Qwen2.5-0.5B 完整跨模型验证")
    print("=" * 60)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  Layers: {n_layers}, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    texts = [
        "The future of artificial intelligence lies in developing more efficient algorithms.",
        "Machine learning has revolutionized how we approach complex problems.",
        "Transformer architecture has become the foundation of modern NLP systems.",
        "Attention mechanisms allow models to focus on relevant input parts.",
        "Large language models show emergent abilities as they scale up.",
    ] * 6

    prompt = "The future of artificial intelligence lies in"
    results = {'model': MODEL, 'n_layers': n_layers}

    # ---- Part A: Degree Ablation (domain=[-10,0]) ----
    print("\n--- Part A: Degree Ablation (domain=[-10,0]) ---")
    DOMAIN = 10.0
    DEGREES = [3, 5, 7, 9, 11]

    ppl_orig = compute_ppl(model, tokenizer, texts, REAL_SOFTMAX, 128, DEVICE)
    print(f"  Baseline PPL: {ppl_orig:.4f}")

    deg_results = []
    for d in DEGREES:
        poly_fn, linf = make_poly_softmax(DOMAIN, d=d)
        ppl = compute_ppl(model, tokenizer, texts, poly_fn, 128, DEVICE)
        cos_sim, top10 = compare_with_real(model, tokenizer, prompt, poly_fn, DEVICE)
        delta = (ppl/ppl_orig - 1) * 100
        print(f"  d={d}: L={linf:.2e}, PPL={ppl:.4f}, delta={delta:+.2f}%, cos={cos_sim:.4f}, top10={top10}/10")
        deg_results.append({'degree': d, 'linf': linf, 'ppl': ppl, 'pct_change': delta, 'cos_sim': cos_sim, 'top10': top10})
    results['degree_ablation'] = {'domain': DOMAIN, 'baseline_ppl': ppl_orig, 'results': deg_results}

    # ---- Part B: Attention KL (d=9, domain=[-10,0]) ----
    print("\n--- Part B: Attention Distribution (d=9, domain=[-10,0]) ---")
    poly_fn, _ = make_poly_softmax(DOMAIN, d=9)
    kl_layers = captured_attention_kl(model, tokenizer, prompt, poly_fn, DEVICE)
    avg_kl = np.mean([l['kl'] for l in kl_layers])
    avg_corr = np.mean([l['pearson'] for l in kl_layers])
    print(f"  Avg KL: {avg_kl:.6f}, Avg Pearson: {avg_corr:.4f}")
    for l in kl_layers:
        if l['layer'] % 4 == 0 or l['layer'] == n_layers - 1:
            print(f"    L{l['layer']:2d}: KL={l['kl']:.6f}, Pearson={l['pearson']:.4f}")
    results['attention_kl'] = {'avg_kl': avg_kl, 'avg_pearson': avg_corr, 'layers': kl_layers}

    # ---- Part C: Domain amplification (d=9) ----
    print("\n--- Part C: Domain Amplification (d=9) ---")
    DOMAINS = [8.0, 10.0, 12.0, 15.0, 20.0]
    dom_results = []
    for M in DOMAINS:
        poly_fn, linf = make_poly_softmax(M, d=9)
        ppl = compute_ppl(model, tokenizer, texts, poly_fn, 128, DEVICE)
        cos_sim, top10 = compare_with_real(model, tokenizer, prompt, poly_fn, DEVICE)
        delta = (ppl/ppl_orig - 1) * 100
        print(f"  [-{M:.0f},0]: L={linf:.2e}, PPL={ppl:.4f}, delta={delta:+.2f}%, cos={cos_sim:.4f}, top10={top10}/10")
        dom_results.append({'domain': M, 'linf': linf, 'ppl': ppl, 'pct_change': delta, 'cos_sim': cos_sim, 'top10': top10})
    results['domain_amplification'] = {'results': dom_results}

    # ---- Cross-model comparison ----
    print("\n--- 跨模型对比: 0.5B vs 7B ---")
    print("  指标                | Qwen2.5-0.5B | Chinese-Llama-2-7b")
    print("  " + "-"*55)
    # Degree ablation
    d5_05b = [r for r in deg_results if r['degree'] == 5][0]
    d9_05b = [r for r in deg_results if r['degree'] == 9][0]
    print(f"  d=5 PPL delta       | {d5_05b['pct_change']:+.2f}%        | +0.11%")
    print(f"  d=9 PPL delta       | {d9_05b['pct_change']:+.2f}%        | -0.14%")
    print(f"  d=5 cos sim         | {d5_05b['cos_sim']:.4f}         | 0.9999")
    print(f"  d=9 cos sim (M=10)  | {d9_05b['cos_sim']:.4f}         | 0.99996")
    print(f"  Attn KL (d=9,M=10)  | {avg_kl:.6f}         | 0.0020")
    print(f"  Attn Pearson        | {avg_corr:.4f}         | 1.0000")

    m8_05b = [r for r in dom_results if r['domain'] == 8.0][0]
    m10_05b = [r for r in dom_results if r['domain'] == 10.0][0]
    print(f"  M=8 PPL delta       | {m8_05b['pct_change']:+.2f}%         | -3.72%")
    print(f"  M=10 PPL delta      | {m10_05b['pct_change']:+.2f}%         | +0.15%")
    print(f"  Domain amplification | {'YES' if abs(m10_05b['pct_change']) < abs(m8_05b['pct_change']) else 'NO'}          | YES")

    output_path = os.path.join(OUTPUT_DIR, 'exp11_05b_full.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")
    print("实验 11 完成！")

if __name__ == '__main__':
    main()
