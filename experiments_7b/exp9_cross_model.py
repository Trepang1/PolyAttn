"""
实验 9: 跨模型验证 (Qwen2.5-0.5B)
=================================
用已有的 0.5B 模型快速验证 domain 放大效应的跨模型一致性。
固定 d=9, 扫 domain {8, 10, 12, 15, 20}
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
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
def compute_ppl_explicit(model, tokenizer, texts, softmax_fn, max_length=128, device='cuda'):
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
    if total_tokens == 0: return float('inf'), float('inf')
    avg = total_loss / total_tokens
    return float(np.exp(avg)), float(avg)

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
    top10 = len(set(orig_top.tolist()) & set(poly_top.tolist()))
    return cos_sim, top10

def main():
    MODEL = '/root/autodl-tmp/Qwen2.5-0.5B'
    DEVICE = 'cuda'
    DEGREE = 9
    DOMAINS = [8.0, 10.0, 12.0, 15.0, 20.0]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"实验 9: 跨模型验证 — {MODEL.split('/')[-1]}")
    print(f"d={DEGREE}, domains={DOMAINS}")
    print("=" * 60)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    texts = [
        "The future of artificial intelligence lies in developing more efficient and scalable algorithms for training and inference.",
        "Machine learning has revolutionized how we approach complex scientific and engineering problems across many disciplines.",
        "Transformer architecture has become the foundation of modern natural language processing systems and beyond.",
        "Attention mechanisms allow models to dynamically focus on the most relevant parts of input sequences during processing.",
        "Large language models demonstrate emergent abilities including in context learning, chain of thought reasoning, and instruction following.",
    ] * 8

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    ppl_orig, _ = compute_ppl_explicit(model, tokenizer, texts, REAL_SOFTMAX, 128, DEVICE)
    print(f"Baseline PPL: {ppl_orig:.4f}")

    results = []
    for M in DOMAINS:
        poly_fn, linf = make_poly_softmax(M, d=DEGREE)
        ppl, _ = compute_ppl_explicit(model, tokenizer, texts, poly_fn, 128, DEVICE)
        delta = ppl - ppl_orig
        pct = (ppl / ppl_orig - 1) * 100
        cos_sim, top10 = compare_with_real(model, tokenizer, prompt, poly_fn, DEVICE)
        print(f"  [-{M:.0f},0]: L={linf:.2e}, PPL={ppl:.4f}, delta={delta:+.4f} ({pct:+.2f}%), cos={cos_sim:.6f}, top10={top10}/10")
        results.append({'domain': M, 'linf_error': linf, 'ppl': ppl, 'ppl_delta': delta, 'ppl_pct_change': pct, 'cos_sim': cos_sim, 'top10_overlap': top10})

    print(f"\n{'Domain':<12} {'L-inf':<12} {'PPL':<10} {'Delta':<10} {'%':<10} {'Cos sim':<10}")
    print("-" * 64)
    for r in results:
        print(f"[-{r['domain']:<4.0f},0]   {r['linf_error']:<12.2e} {r['ppl']:<10.4f} {r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.2f}% {r['cos_sim']:<10.6f}")

    # Compare with 7B results
    print("\n--- 与 Chinese-Llama-2-7b 对比 ---")
    print(f"  Model: 0.5B (Qwen2.5) vs 7B (Chinese-Llama-2)")
    print(f"  M=8  delta: {results[0]['ppl_pct_change']:+.2f}% (0.5B) vs -3.72% (7B)")
    print(f"  M=10 delta: {results[1]['ppl_pct_change']:+.2f}% (0.5B) vs +0.15% (7B)")
    print(f"  M=20 delta: {results[-1]['ppl_pct_change']:+.2f}% (0.5B) vs +0.63% (7B)")

    output = {'experiment': 'cross_model_05b', 'model': MODEL, 'degree': DEGREE, 'baseline_ppl': ppl_orig, 'results': results}
    out_path = os.path.join(OUTPUT_DIR, 'exp9_cross_model_05b.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
