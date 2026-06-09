"""
实验 1: Degree 消融实验
========================
固定 domain=[-8,0], 扫描 d=3,5,7,9,11,13
目标: 证明 d=9 是精度饱和的最小 degree
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Polynomial utilities (same as polyattn_ppl_test.py)
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
    n_mul = d
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
    return fn, linf, n_mul, cn

# ============================================================================
# PPL computation
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

# ============================================================================
# Output comparison
# ============================================================================

# Store the REAL original softmax at import time
REAL_SOFTMAX = F.softmax

@torch.no_grad()
def compare_outputs(model, tokenizer, prompt, poly_softmax_fn, device='cuda'):
    """Compare original (exp) vs PolyAttn outputs. Uses REAL_SOFTMAX as baseline."""
    import torch.nn.functional as F_ref
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)

    # Always use the real exp softmax for baseline
    F_ref.softmax = REAL_SOFTMAX
    out_orig = model(input_ids, output_hidden_states=True)

    # PolyAttn
    F_ref.softmax = poly_softmax_fn
    try:
        out_poly = model(input_ids, output_hidden_states=True)
    finally:
        F_ref.softmax = REAL_SOFTMAX  # restore to real softmax

    cos_sim = float(F_ref.cosine_similarity(
        out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    logits_max_diff = float((out_orig.logits.float() - out_poly.logits.float()).abs().max().item())
    logits_mean_diff = float((out_orig.logits.float() - out_poly.logits.float()).abs().mean().item())
    _, orig_top = out_orig.logits[0, -1].topk(10)
    _, poly_top = out_poly.logits[0, -1].topk(10)
    topk_overlap = len(set(orig_top.tolist()) & set(poly_top.tolist()))
    return {
        'cos_sim': cos_sim,
        'logits_max_diff': logits_max_diff,
        'logits_mean_diff': logits_mean_diff,
        'top10_overlap': topk_overlap,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DOMAIN_M = 8.0
    DEGREES = [3, 5, 7, 9, 11, 13]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("实验 1: Degree 消融实验")
    print(f"模型: {MODEL}")
    print(f"固定 domain: [-{DOMAIN_M}, 0]")
    print(f"扫描 degree: {DEGREES}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB used / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB total")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Test texts
    texts = [
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",
        "深度学习的发展推动了自然语言处理领域的巨大进步，从机器翻译到文本生成都取得了突破性成果。",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",
        "计算机视觉技术使得机器能够理解和分析图像内容，广泛应用于自动驾驶、医疗诊断等领域。",
        "量子计算作为一种新型计算范式，有望在密码学、药物研发等领域带来革命性变化。",
        "The future of artificial intelligence lies in developing more efficient algorithms.",
        "Machine learning has revolutionized how we approach complex problems in science.",
        "Natural language processing enables computers to understand human language at scale.",
        "Deep neural networks have achieved remarkable results across many domains.",
        "Privacy-preserving machine learning is critical for deploying AI in sensitive applications.",
        "中国在人工智能领域的研究投入不断增加，推动了技术创新。",
        "自然语言处理是人工智能的重要分支，涵盖了多个方向。",
        "大数据时代，如何有效保护用户隐私成为了关键挑战。",
        "机器学习算法在图像识别、语音识别等任务上已经超过了人类水平。",
        "强化学习在游戏对弈和机器人控制领域展现出了强大的能力。",
        "Transformer architecture has become the foundation of modern NLP systems.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Transfer learning has dramatically reduced the data requirements for NLP tasks.",
        "Large language models demonstrate emergent abilities as they scale up in size.",
        "The combination of scale and data quality determines model performance.",
    ] * 2  # 40 texts

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    # Baseline
    print("\n--- Baseline (original exp Softmax) ---")
    orig_softmax = F.softmax
    t0 = time.time()
    ppl_orig, loss_orig = compute_ppl(model, tokenizer, texts, max_length=128, device=DEVICE)
    dt_orig = time.time() - t0
    print(f"  PPL={ppl_orig:.4f}, loss={loss_orig:.4f}, time={dt_orig:.1f}s")

    # Degree sweep
    results = []
    for d in DEGREES:
        print(f"\n--- d={d} ---")
        poly_fn, linf, n_mul, coeffs = make_poly_softmax(DOMAIN_M, d=d)
        print(f"  L∞ error: {linf:.2e}")
        print(f"  Multiplications per element: {n_mul}")

        # PPL
        F.softmax = poly_fn
        t0 = time.time()
        ppl, loss = compute_ppl(model, tokenizer, texts, max_length=128, device=DEVICE)
        dt = time.time() - t0
        delta_ppl = ppl - ppl_orig
        pct = (ppl / ppl_orig - 1) * 100
        print(f"  PPL={ppl:.4f}, delta={delta_ppl:+.4f} ({pct:+.3f}%), time={dt:.1f}s")

        # Output comparison
        comp = compare_outputs(model, tokenizer, prompt, poly_fn, DEVICE)
        print(f"  Cos sim: {comp['cos_sim']:.6f}")
        print(f"  Logits max diff: {comp['logits_max_diff']:.6e}")
        print(f"  Logits mean diff: {comp['logits_mean_diff']:.6e}")
        print(f"  Top-10 overlap: {comp['top10_overlap']}/10")

        results.append({
            'degree': d,
            'linf_error': linf,
            'num_multiplications': n_mul,
            'ppl': ppl,
            'ppl_delta': delta_ppl,
            'ppl_pct_change': pct,
            'loss': loss,
            'cos_sim': comp['cos_sim'],
            'logits_max_diff': comp['logits_max_diff'],
            'logits_mean_diff': comp['logits_mean_diff'],
            'top10_overlap': comp['top10_overlap'],
            'inference_time_s': dt,
            'coeffs': [float(c) for c in coeffs],
        })

    F.softmax = orig_softmax

    # Summary
    print("\n" + "=" * 70)
    print("Degree 消融实验结果汇总")
    print("=" * 70)
    header = f"{'d':<6} {'L∞':<12} {'PPL':<10} {'ΔPPL':<10} {'Δ%':<10} {'CosSim':<10} {'Top10':<8} {'Mul':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['degree']:<6} {r['linf_error']:<12.2e} {r['ppl']:<10.4f} "
              f"{r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.3f}% "
              f"{r['cos_sim']:<10.6f} {r['top10_overlap']:<8}/10 {r['num_multiplications']:<6}")

    # Best degree
    best = min(results, key=lambda r: abs(r['ppl_delta']))
    print(f"\n最优 degree: d={best['degree']} (PPL delta = {best['ppl_delta']:+.4f})")

    # Check saturation
    # Find smallest d where PPL delta < 0.01%
    saturated = [r for r in results if abs(r['ppl_pct_change']) < 0.01]
    if saturated:
        smallest_sat = min(saturated, key=lambda r: r['degree'])
        print(f"精度饱和点: d={smallest_sat['degree']} (|ΔPPL%| < 0.01%)")
        print(f"  → d=9 vs d={smallest_sat['degree']}: {'更优' if 9 >= smallest_sat['degree'] else '不够'}")

    # Save
    output = {
        'experiment': 'degree_ablation',
        'model': MODEL,
        'fixed_domain': DOMAIN_M,
        'baseline_ppl': ppl_orig,
        'baseline_loss': loss_orig,
        'results': results,
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp1_degree_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")

    # Markdown table for paper
    print("\n--- LaTeX-ready table ---")
    print("Degree & L∞ error & PPL & ΔPPL & Δ% & Cos sim & Top-10 \\\\ \\hline")
    for r in results:
        print(f"d={r['degree']} & {r['linf_error']:.2e} & {r['ppl']:.2f} & "
              f"{r['ppl_delta']:+.2f} & {r['ppl_pct_change']:+.2f}\\% & "
              f"{r['cos_sim']:.4f} & {r['top10_overlap']}/10 \\\\")

    print("\n实验 1 完成！")
    return results

if __name__ == '__main__':
    main()
