"""
实验 2: Domain × Degree 联合扫描
=================================
3×5 网格: d ∈ {7, 9, 11} × domain ∈ {6, 8, 10, 12, 15}
目标: 确认 d=9 在哪些 domain 下精度饱和，找出最优 (d, domain) 组合
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
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
    cheb[0] /= 2.0
    cheb = cheb[:d + 1]
    T = np.zeros((d + 1, d + 1))
    T[0, 0] = 1.0
    if d >= 1: T[1, 1] = 1.0
    for k_idx in range(2, d + 1):
        T[k_idx, 1:] += 2.0 * T[k_idx - 1, :-1]
        T[k_idx, :] -= T[k_idx - 2, :]
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
    f_true = np.exp(-lam * grid)
    p_vals = np.polyval(coeffs[::-1], grid)
    return float(np.max(np.abs(f_true - p_vals)))

def horner_torch(c, x):
    y = torch.full_like(x, c[-1])
    for ci in reversed(c[:-1]): y = y * x + ci
    return y

def make_poly_softmax(M_domain, d=9):
    lam = 1.0
    coeffs = generate_polynomial(lam, M_domain, d)
    linf = check_approx(coeffs, lam, M_domain)
    cn = np.array(coeffs, dtype=np.float64)
    def fn(logits, dim=-1, dtype=None, **kw):
        x = logits.float()
        mx = x.max(dim=dim, keepdim=True).values
        xs = x - mx
        xc = torch.clamp(xs, -M_domain, 0.0)
        y = horner_torch(cn, -xc)
        y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return fn, linf

# ============================================================================
# PPL
# ============================================================================

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, max_length=128, device='cuda'):
    model.eval()
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
        except Exception as e:
            pass
    if total_tokens == 0: return float('inf'), float('inf')
    avg = total_loss / total_tokens
    return float(np.exp(avg)), float(avg)

REAL_SOFTMAX_EXP2 = F.softmax  # Store real exp softmax

@torch.no_grad()
def compare_outputs(model, tokenizer, prompt, poly_softmax_fn, device='cuda'):
    """Compare original (exp) vs PolyAttn outputs."""
    import torch.nn.functional as F_ref
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)
    # Baseline = real exp softmax
    F_ref.softmax = REAL_SOFTMAX_EXP2
    out_orig = model(input_ids, output_hidden_states=True)
    F_ref.softmax = poly_softmax_fn
    try:
        out_poly = model(input_ids, output_hidden_states=True)
    finally:
        F_ref.softmax = REAL_SOFTMAX_EXP2
    cos_sim = float(F_ref.cosine_similarity(
        out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    _, orig_top = out_orig.logits[0, -1].topk(10)
    _, poly_top = out_poly.logits[0, -1].topk(10)
    topk_overlap = len(set(orig_top.tolist()) & set(poly_top.tolist()))
    return {'cos_sim': cos_sim, 'top10_overlap': topk_overlap}

# ============================================================================
# Main
# ============================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DEGREES = [7, 9, 11]
    DOMAINS = [6.0, 8.0, 10.0, 12.0, 15.0]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("实验 2: Domain × Degree 联合扫描")
    print(f"Degree: {DEGREES}")
    print(f"Domain: {DOMAINS}")
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

    texts = [
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",
        "深度学习的发展推动了自然语言处理领域的巨大进步。",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",
        "计算机视觉技术使得机器能够理解和分析图像内容。",
        "量子计算作为一种新型计算范式，有望在密码学等领域带来革命性变化。",
        "The future of artificial intelligence lies in developing more efficient algorithms.",
        "Machine learning has revolutionized how we approach complex problems in science.",
        "Natural language processing enables computers to understand human language at scale.",
        "中国在人工智能领域的研究投入不断增加，推动了技术创新。",
        "大数据时代，如何有效保护用户隐私成为了关键挑战。",
        "Transformer architecture has become the foundation of modern NLP systems.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models demonstrate emergent abilities as they scale up.",
        "Privacy-preserving machine learning is critical for deploying AI.",
        "Deep neural networks have achieved remarkable results across many domains.",
    ] * 3  # 45 texts

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    # Baseline
    orig_softmax = F.softmax
    ppl_orig, loss_orig = compute_ppl(model, tokenizer, texts, max_length=128, device=DEVICE)
    print(f"\nBaseline PPL: {ppl_orig:.4f} (loss={loss_orig:.4f})")

    # Grid sweep
    results = []
    total = len(DEGREES) * len(DOMAINS)
    idx = 0
    for d in DEGREES:
        for dm in DOMAINS:
            idx += 1
            print(f"\n[{idx}/{total}] d={d}, domain=[-{dm},0]")
            poly_fn, linf = make_poly_softmax(dm, d=d)

            F.softmax = poly_fn
            t0 = time.time()
            ppl, loss = compute_ppl(model, tokenizer, texts, max_length=128, device=DEVICE)
            dt = time.time() - t0
            delta = ppl - ppl_orig
            pct = (ppl / ppl_orig - 1) * 100

            comp = compare_outputs(model, tokenizer, prompt, poly_fn, DEVICE)

            print(f"  L∞={linf:.2e}, PPL={ppl:.4f}, Δ={delta:+.4f} ({pct:+.3f}%), "
                  f"cos={comp['cos_sim']:.5f}, top10={comp['top10_overlap']}/10, t={dt:.0f}s")

            results.append({
                'degree': d, 'domain': dm, 'linf_error': linf,
                'ppl': ppl, 'ppl_delta': delta, 'ppl_pct_change': pct,
                'cos_sim': comp['cos_sim'], 'top10_overlap': comp['top10_overlap'],
                'inference_time_s': dt,
            })

    F.softmax = orig_softmax

    # Summary
    print("\n" + "=" * 80)
    print("Domain × Degree 联合扫描结果")
    print("=" * 80)
    print(f"{'d\\Domain':<8}", end="")
    for dm in DOMAINS:
        print(f"[-{dm:<4.0f},0]  ", end="")
    print()
    print("-" * 80)

    for d in DEGREES:
        print(f"{'d='+str(d):<8}", end="")
        for dm in DOMAINS:
            r = next(rr for rr in results if rr['degree'] == d and rr['domain'] == dm)
            print(f"Δ={r['ppl_delta']:+.4f}   ", end="")
        print()

    print(f"\n{'d\\Domain':<8}", end="")
    for dm in DOMAINS:
        print(f"[-{dm:<4.0f},0]    ", end="")
    print()
    for d in DEGREES:
        print(f"{'d='+str(d):<8}", end="")
        for dm in DOMAINS:
            r = next(rr for rr in results if rr['degree'] == d and rr['domain'] == dm)
            print(f"cos={r['cos_sim']:.4f}  ", end="")
        print()

    # Best by PPL
    best_ppl = min(results, key=lambda r: abs(r['ppl_delta']))
    print(f"\n最优 (PPL): d={best_ppl['degree']}, domain=[-{best_ppl['domain']},0], "
          f"ΔPPL={best_ppl['ppl_delta']:+.4f}")

    # Pareto-optimal: smallest (d, domain) with |ΔPPL| < 0.01%
    pareto = [r for r in results if abs(r['ppl_pct_change']) < 0.01]
    if pareto:
        best = min(pareto, key=lambda r: (r['degree'], r['domain']))
        print(f"Pareto 最优 (|Δ%| < 0.01%): d={best['degree']}, domain=[-{best['domain']},0]")
        print(f"  → 最省计算且精度无损的配置")

    # Save
    output = {
        'experiment': 'domain_degree_sweep',
        'model': MODEL,
        'degrees': DEGREES,
        'domains': DOMAINS,
        'baseline_ppl': ppl_orig,
        'baseline_loss': loss_orig,
        'results': results,
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp2_domain_degree_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    print("实验 2 完成！")

if __name__ == '__main__':
    main()
