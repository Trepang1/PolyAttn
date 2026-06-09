"""
实验 10: 13B Domain 放大 + Degree 消融
=======================================
在 Chinese-Llama-2-13b 上复现最关键的实验 7 (domain amplification) + 实验 1 (degree ablation)
VRAM 紧张 (~26GB)，使用较小的文本集和序列长度
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
def compute_ppl_explicit(model, tokenizer, texts, softmax_fn, max_length=64, device='cuda'):
    """Minimal VRAM: short max_length=64"""
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
        except torch.cuda.OutOfMemoryError:
            print("  OOM! Trying shorter...")
            torch.cuda.empty_cache()
            continue
        except: pass
    if total_tokens == 0: return float('inf'), float('inf')
    avg = total_loss / total_tokens
    return float(np.exp(avg)), float(avg)

@torch.no_grad()
def compare_with_real(model, tokenizer, prompt, poly_fn, device='cuda'):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
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

def find_model_dir():
    """Find downloaded 13B model directory."""
    import glob
    candidates = [
        '/root/autodl-tmp/chinese-llama-2-13b',
        '/root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Search
    matches = glob.glob('/root/autodl-tmp/**/chinese*llama*13b*', recursive=True)
    if matches:
        return matches[0]
    matches = glob.glob('/root/autodl-tmp/**/config.json', recursive=True)
    for m in matches:
        d = os.path.dirname(m)
        if '13b' in d.lower() or '13B' in d:
            return d
    return None

def main():
    DEVICE = 'cuda'
    DEGREE = 9
    DOMAINS = [8.0, 10.0, 12.0, 15.0, 20.0]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find model
    MODEL = find_model_dir()
    if not MODEL:
        print("ERROR: 13B model not found!")
        print("Available dirs:")
        for d in os.listdir('/root/autodl-tmp/'):
            print(f"  {d}")
        return

    print("=" * 60)
    print(f"实验 10: 13B Domain 放大")
    print(f"Model: {MODEL}")
    print(f"d={DEGREE}, domains={DOMAINS}")
    print(f"VRAM free: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated()/1e9:.1f}GB")
    print("=" * 60)

    print("\nLoading model (bf16, this is tight)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  {n_params:.1f}B params, VRAM: {vram:.1f}/{vram_total:.1f}GB")

    # Short texts to save VRAM
    texts = [
        "The future of artificial intelligence lies in developing more efficient algorithms.",
        "Machine learning has revolutionized how we approach complex problems in science.",
        "Transformer architecture has become the foundation of modern NLP systems.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "人工智能技术正在快速发展，大语言模型成为了最受关注的研究方向。",
        "深度学习推动了自然语言处理领域的巨大进步和突破。",
        "在数字化时代，数据安全和隐私保护成为重要研究课题。",
        "计算机视觉技术使机器能够理解和分析图像内容。",
    ] * 3

    prompt = "The future of artificial intelligence lies in"

    # Baseline
    print("\n--- Baseline ---")
    ppl_orig, _ = compute_ppl_explicit(model, tokenizer, texts, REAL_SOFTMAX, 64, DEVICE)
    print(f"  PPL={ppl_orig:.4f}")

    results = []
    for M in DOMAINS:
        torch.cuda.empty_cache()
        print(f"\n--- Domain=[-{M:.0f},0] ---")
        poly_fn, linf = make_poly_softmax(M, d=DEGREE)
        print(f"  L-inf={linf:.2e}")

        ppl, _ = compute_ppl_explicit(model, tokenizer, texts, poly_fn, 64, DEVICE)
        delta = ppl - ppl_orig
        pct = (ppl / ppl_orig - 1) * 100
        print(f"  PPL={ppl:.4f}, delta={delta:+.4f} ({pct:+.2f}%)")

        cos_sim, top10 = compare_with_real(model, tokenizer, prompt, poly_fn, DEVICE)
        print(f"  cos={cos_sim:.6f}, top10={top10}/10")

        results.append({
            'domain': M, 'linf_error': linf,
            'ppl': ppl, 'ppl_delta': delta, 'ppl_pct_change': pct,
            'cos_sim': cos_sim, 'top10_overlap': top10,
        })

    # Summary
    print(f"\n{'Domain':<12} {'PPL':<10} {'Delta':<10} {'%':<10} {'Cos sim':<10}")
    print("-" * 52)
    for r in results:
        print(f"[-{r['domain']:<4.0f},0]   {r['ppl']:<10.4f} {r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.2f}% {r['cos_sim']:<10.6f}")

    # Compare with 7B
    print("\n--- 13B vs 7B 对比 ---")
    # 7B reference values from exp7
    ref_7b = {8.0: -3.72, 10.0: 0.15, 12.0: 0.17, 15.0: 0.09, 20.0: 0.63}
    for r in results:
        m = r['domain']
        ref = ref_7b.get(m, 0)
        print(f"  M={m:.0f}: 13B delta={r['ppl_pct_change']:+.2f}%, 7B delta={ref:+.2f}%")

    output = {
        'experiment': '13b_domain_amplification',
        'model': MODEL, 'degree': DEGREE,
        'n_params': n_params, 'vram_gb': vram,
        'baseline_ppl': ppl_orig,
        'results': results,
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp10_13b_domain.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
