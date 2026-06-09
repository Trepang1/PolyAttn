"""
实验 7: Domain 放大实验 — 分离 clip 效应 vs 多项式效应
========================================================
核心问题: PPL 下降 2-9% 是 clip 造成的还是多项式逼近不准造成的？

方法: 固定 d=9, 扫描 domain M ∈ {8, 10, 12, 15, 20, 25}
- M 小 → clip 效应强 → 但如果多项式准, PPL 也不该大变
- M 大 → clip 效应≈0 → 纯看多项式效应
- 如果 PPL 随 M 增大而回归基线 → clip 是真凶, 多项式没问题
- 如果 PPL 不管 M 大小都下降 → 多项式本身有问题
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Store REAL softmax at import time
# ============================================================================
REAL_SOFTMAX = F.softmax

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

# ============================================================================
# PPL with explicit softmax
# ============================================================================

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

# ============================================================================
# Output comparison with explicit softmax
# ============================================================================

@torch.no_grad()
def compare_with_real(model, tokenizer, prompt, poly_fn, device='cuda'):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)

    F.softmax = REAL_SOFTMAX
    out_orig = model(input_ids)

    F.softmax = poly_fn
    out_poly = model(input_ids)

    F.softmax = REAL_SOFTMAX

    cos_sim = float(F.cosine_similarity(
        out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    diff = float((out_orig.logits.float() - out_poly.logits.float()).abs().max().item())
    _, orig_top = out_orig.logits[0, -1].topk(10)
    _, poly_top = out_poly.logits[0, -1].topk(10)
    top10 = len(set(orig_top.tolist()) & set(poly_top.tolist()))
    return cos_sim, diff, top10

# ============================================================================
# Main
# ============================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DEGREE = 9
    DOMAINS = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("实验 7: Domain 放大 — 分离 clip 效应 vs 多项式效应")
    print(f"固定: d={DEGREE}")
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

    # Texts
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
        "强化学习在游戏对弈和机器人控制领域展现了强大的自主学习能力。",
        "预训练加微调范式极大降低了自然语言处理任务对标注数据的需求。",
        "模型量化技术通过降低参数精度来减少模型存储和计算开销。",
        "多模态学习结合文本图像语音等多种信息模态提升模型理解能力。",
        "图神经网络扩展深度学习到非欧几里得结构的图数据上。",
    ] * 3  # 45 texts

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    # Baseline
    print("\n--- Baseline (REAL exp softmax) ---")
    ppl_orig, loss_orig = compute_ppl_explicit(model, tokenizer, texts, REAL_SOFTMAX, 128, DEVICE)
    print(f"  PPL={ppl_orig:.4f}")

    # Domain sweep
    results = []
    for M in DOMAINS:
        print(f"\n--- Domain=[-{M:.0f},0], d={DEGREE} ---")
        poly_fn, linf = make_poly_softmax(M, d=DEGREE)
        print(f"  L-inf error: {linf:.2e}")

        # PPL
        ppl, loss = compute_ppl_explicit(model, tokenizer, texts, poly_fn, 128, DEVICE)
        delta = ppl - ppl_orig
        pct = (ppl / ppl_orig - 1) * 100

        # Cos sim
        cos_sim, max_diff, top10 = compare_with_real(model, tokenizer, prompt, poly_fn, DEVICE)

        print(f"  PPL={ppl:.4f}, delta={delta:+.4f} ({pct:+.2f}%)")
        print(f"  Cos sim={cos_sim:.6f}, max diff={max_diff:.4f}, top-10={top10}/10")

        results.append({
            'domain': M, 'linf_error': linf,
            'ppl': ppl, 'ppl_delta': delta, 'ppl_pct_change': pct,
            'cos_sim': cos_sim, 'logits_max_diff': max_diff, 'top10_overlap': top10,
        })

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Domain 放大实验结果")
    print(f"基线 PPL: {ppl_orig:.4f}")
    print("=" * 70)
    print(f"{'Domain':<12} {'L-inf':<12} {'PPL':<10} {'Delta':<10} {'%':<10} {'Cos sim':<10} {'Top10':<8}")
    print("-" * 72)
    for r in results:
        print(f"[-{r['domain']:<4.0f},0]   {r['linf_error']:<12.2e} {r['ppl']:<10.4f} "
              f"{r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.2f}% "
              f"{r['cos_sim']:<10.6f} {r['top10_overlap']:<8}/10")

    # ---- Analysis ----
    print("\n--- 趋势分析 ---")
    deltas = [r['ppl_delta'] for r in results]
    cos_sims = [r['cos_sim'] for r in results]
    linfs = [r['linf_error'] for r in results]

    # Is delta approaching 0?
    min_abs_delta = min(results, key=lambda r: abs(r['ppl_delta']))
    print(f"最接近基线的 domain: [-{min_abs_delta['domain']:.0f},0], delta={min_abs_delta['ppl_delta']:+.4f}")

    # Cos sim trend
    best_cos = max(results, key=lambda r: r['cos_sim'])
    print(f"最高 cos sim 的 domain: [-{best_cos['domain']:.0f},0], cos={best_cos['cos_sim']:.6f}")

    # Key conclusion
    if abs(deltas[-1]) < abs(deltas[0]):
        print(f"\n>> PPL 随 domain 扩大而回归基线: M=8 delta={deltas[0]:+.2f}% → M={DOMAINS[-1]} delta={deltas[-1]:+.2f}%")
        print(">> 结论: PPL 变化主要来自 clip 效应, 多项式逼近本身对模型影响有限")
    else:
        print(f"\n>> PPL 不随 domain 扩大而回归: M=8 delta={deltas[0]:+.2f}% → M={DOMAINS[-1]} delta={deltas[-1]:+.2f}%")
        print(">> 结论: PPL 变化来自多项式逼近本身, 需重新审视方法的 fidelity")

    # Save
    output = {
        'experiment': 'domain_amplification',
        'model': MODEL, 'degree': DEGREE,
        'baseline_ppl': ppl_orig,
        'results': results,
        'conclusion': 'clip_dominant' if abs(deltas[-1]) < abs(deltas[0]) else 'poly_dominant',
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp7_domain_amplification.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")

if __name__ == '__main__':
    main()
