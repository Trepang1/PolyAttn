"""
实验 8: d=5 最优度验证
======================
d=5 在实验 1 中 cos_sim=0.9999, ΔPPL=+0.11% (最接近原模型)
在多个 domain 下验证 d=5 的鲁棒性
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
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DEGREE = 5
    DOMAINS = [8.0, 10.0, 12.0, 15.0, 20.0]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"实验 8: d={DEGREE} 多 domain 验证 (经验最优 degree)")
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
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。深度学习推动了自然语言处理的巨大进步。",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。计算机视觉技术使机器能理解和分析图像内容。",
        "量子计算有望在密码学和药物研发领域带来革命性变化。中国在人工智能领域的研究投入不断增加。",
        "自然语言处理是人工智能的重要分支，涵盖了文本分类、情感分析等多个方向。",
        "大数据时代如何有效保护用户隐私成为了关键挑战。联邦学习能在保护隐私的同时实现协同训练。",
        "The future of artificial intelligence lies in developing more efficient and scalable algorithms.",
        "Machine learning has revolutionized how we approach complex scientific and engineering problems.",
        "Transformer architecture has become the foundation of modern natural language processing systems.",
        "Attention mechanisms allow models to dynamically focus on the most relevant parts of input sequences.",
        "Large language models demonstrate emergent abilities including in context learning and chain of thought reasoning.",
    ] * 4

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    # Baseline
    ppl_orig, loss_orig = compute_ppl_explicit(model, tokenizer, texts, REAL_SOFTMAX, 128, DEVICE)
    print(f"Baseline PPL: {ppl_orig:.4f}")

    results = []
    for M in DOMAINS:
        poly_fn, linf = make_poly_softmax(M, d=DEGREE)
        ppl, loss = compute_ppl_explicit(model, tokenizer, texts, poly_fn, 128, DEVICE)
        delta = ppl - ppl_orig
        pct = (ppl / ppl_orig - 1) * 100
        cos_sim, top10 = compare_with_real(model, tokenizer, prompt, poly_fn, DEVICE)
        print(f"  [-{M:.0f},0]: L-inf={linf:.2e}, PPL={ppl:.4f}, delta={delta:+.4f} ({pct:+.2f}%), cos={cos_sim:.6f}, top10={top10}/10")
        results.append({'domain': M, 'linf_error': linf, 'ppl': ppl, 'ppl_delta': delta, 'ppl_pct_change': pct, 'cos_sim': cos_sim, 'top10_overlap': top10})

    print(f"\n{'Domain':<12} {'L-inf':<12} {'PPL':<10} {'Delta':<10} {'%':<10} {'Cos sim':<10} {'Top10':<8}")
    print("-" * 72)
    for r in results:
        print(f"[-{r['domain']:<4.0f},0]   {r['linf_error']:<12.2e} {r['ppl']:<10.4f} {r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.2f}% {r['cos_sim']:<10.6f} {r['top10_overlap']:<8}/10")

    output = {'experiment': 'd5_benchmark', 'model': MODEL, 'degree': DEGREE, 'baseline_ppl': ppl_orig, 'results': results}
    out_path = os.path.join(OUTPUT_DIR, 'exp8_d5_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
