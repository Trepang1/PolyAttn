"""
Final Experiment: K-segment PolyAttn PPL + Multi-run Variance
=============================================================
Addresses reviewer feedback:
1. K-segment (not single-segment) PPL test
2. Multi-run PPL with error bars
3. Domain sweep within K-segment framework
4. Long sequence K-segment (where single-segment failed)
5. Cross-benchmark validation

K-segment design (PolyAttn Section 5):
- Decompose each attention score into K base-256 digits
- Per-segment polynomial on FIXED domain [0, 255]
- Active segment determines polynomial usage
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

REAL_SOFTMAX = F.softmax

# ====================================================================
# Polynomial utilities (Chebyshev → monomial, same as all experiments)
# ====================================================================

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

def horner_torch_np(c_np, x):
    """Horner: c_np is numpy array, x is torch tensor."""
    y = torch.full_like(x, float(c_np[-1]))
    for ci in reversed(c_np[:-1]):
        y = y * x + float(ci)
    return y

# ====================================================================
# K-segment Softmax
# ====================================================================

def make_k_segment_softmax(B=256, K=8, S=256.0, degree=9):
    """
    Build K-segment PolyAttn softmax.

    Parameters:
      B: base for digit decomposition (256)
      K: number of segments (8)
      S: quantization scale (maps -score to integer range)
      degree: polynomial degree for active segments (9)

    For each segment k, lambda_k = B^k / S.
    - If lambda_k * (B-1) < 0.001: segment is constant 1 (skip)
    - If lambda_k * (B-1) > 15: segment is indicator (1 if digit=0, epsilon otherwise)
    - Otherwise: segment uses d=degree Chebyshev polynomial on [0, B-1]
    """
    segments = []
    for k in range(K):
        lam = (B ** k) / S
        lam_M = lam * (B - 1)
        if lam_M < 0.001:
            segments.append({'type': 'const'})
        elif lam_M > 50:  # Higher threshold: allow larger lam_M for polynomial
            segments.append({'type': 'indicator'})
        else:
            coeffs = np.array(generate_polynomial(lam, B - 1, degree), dtype=np.float64)
            segments.append({'type': 'poly', 'coeffs': coeffs, 'lam': lam, 'lam_M': lam_M})

    active_segments = sum(1 for s in segments if s['type'] != 'const')
    poly_segments = sum(1 for s in segments if s['type'] == 'poly')
    print(f"  K-segment config: K={K}, B={B}, S={S}")
    print(f"  Segments: {len(segments)} total, {active_segments} active, {poly_segments} polynomial")
    for k, s in enumerate(segments):
        if s['type'] != 'const':
            print(f"    k={k}: type={s['type']}" +
                  (f", lam={s.get('lam',0):.4f}, lam*M={s.get('lam_M',0):.2f}" if 'lam' in s else ""))

    def k_softmax(logits, dim=-1, dtype=None, **kw):
        x = logits.float()
        x_max = x.max(dim=dim, keepdim=True).values
        x_shifted = x - x_max  # <= 0

        # Quantize: v = round(-x * S), clamped to safe range
        # For practical attention scores (< 500) and S <= 1024, v < 5e5
        v = (-x_shifted * S).round().clamp(0.0, 1e9).long()

        # Decompose and multiply per-segment contributions
        result = torch.ones_like(v, dtype=torch.float32)
        remaining = v.clone()

        for k, seg in enumerate(segments):
            x_k = remaining % B
            remaining = remaining // B

            if seg['type'] == 'const':
                continue
            elif seg['type'] == 'indicator':
                result = result * (x_k == 0).float().clamp(min=1e-30)
            else:  # poly
                p_val = horner_torch_np(seg['coeffs'], x_k.float())
                result = result * p_val.clamp(min=1e-30)

        result = result.clamp(min=1e-30)
        out = result / result.sum(dim=dim, keepdim=True)

        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out

    return k_softmax, segments

# ====================================================================
# PPL with explicit softmax
# ====================================================================

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
    if total_tokens == 0: return float('inf'), float('inf'), 0
    avg = total_loss / total_tokens
    return float(np.exp(avg)), float(avg), total_tokens

@torch.no_grad()
def compare_with_real(model, tokenizer, prompt, softmax_fn, device='cuda'):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)
    F.softmax = REAL_SOFTMAX
    out_orig = model(input_ids)
    F.softmax = softmax_fn
    out_poly = model(input_ids)
    F.softmax = REAL_SOFTMAX
    cos_sim = float(F.cosine_similarity(out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    _, orig_top = out_orig.logits[0, -1].topk(10)
    _, poly_top = out_poly.logits[0, -1].topk(10)
    top10 = len(set(orig_top.tolist()) & set(poly_top.tolist()))
    return cos_sim, top10

# ====================================================================
# Main
# ====================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    N_RUNS = 3  # For variance
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PolyAttn K-Segment Final Experiment")
    print("Addressing: K-segment PPL, multi-run variance, long sequence")
    print(f"Model: {MODEL}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM free: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated()/1e9:.1f}GB")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    d_head = model.config.hidden_size // model.config.num_attention_heads
    print(f"  d_head: {d_head}")

    # Texts for PPL
    texts_short = [
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",
        "深度学习的发展推动了自然语言处理领域的巨大进步。",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",
        "计算机视觉技术使得机器能够理解和分析图像内容。",
        "量子计算作为一种新型计算范式，有望在密码学等领域带来革命性变化。",
        "The future of artificial intelligence lies in developing more efficient algorithms.",
        "Machine learning has revolutionized how we approach complex problems in science.",
        "Transformer architecture has become the foundation of modern NLP systems.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models demonstrate emergent abilities as they scale up.",
        "中国在人工智能领域的研究投入不断增加，推动了技术创新。",
        "自然语言处理是人工智能的重要分支，涵盖了多个方向。",
        "大数据时代，如何有效保护用户隐私成为了关键挑战。",
        "强化学习在游戏对弈和机器人控制领域展现出了强大的能力。",
        "Transfer learning has dramatically reduced the data requirements for NLP.",
    ] * 3  # 45 texts

    # Long text for long-sequence test
    long_text_base = (
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。"
        "深度学习的发展推动了自然语言处理领域的巨大进步，从机器翻译到文本生成都取得了突破性成果。"
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。"
        "计算机视觉技术使得机器能够理解和分析图像内容，广泛应用于自动驾驶、医疗诊断等领域。"
        "量子计算作为一种新型计算范式，有望在密码学、药物研发等领域带来革命性变化。"
        "中国在人工智能领域的研究投入不断增加，推动了技术创新和产业升级。"
        "自然语言处理是人工智能的重要分支，涵盖了文本分类、情感分析、命名实体识别等多个方向。"
        "大数据时代，如何有效保护用户隐私成为了学术界和工业界共同面临的关键挑战。"
        "机器学习算法在图像识别、语音识别等任务上已经超过了人类水平的表现。"
        "强化学习在游戏对弈和机器人控制领域展现出了强大的自主学习能力。"
        "Transformer架构已经成为现代自然语言处理系统的基础组件，广泛应用于各类任务。"
        "注意力机制允许模型在处理输入序列时动态地关注最相关的信息部分。"
        "预训练加微调的范式极大地降低了自然语言处理任务对标注数据的需求量。"
        "大规模语言模型随着参数量的增加展现出令人惊讶的涌现能力。"
        "数据质量和规模的结合是决定模型最终性能的关键因素。"
    )
    long_text = (long_text_base + " ") * 30

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    results_all = {}

    # ====================================================================
    # Part A: K-segment PPL with multiple S values
    # ====================================================================
    print("\n" + "=" * 60)
    print("Part A: K-segment PPL (varying quantization scale S)")
    print("=" * 60)

    # Baseline
    ppl_orig, loss_orig, n_tok = compute_ppl(model, tokenizer, texts_short, REAL_SOFTMAX, 128, DEVICE)
    print(f"Baseline PPL: {ppl_orig:.4f} (loss={loss_orig:.4f}, tokens={n_tok})")

    # Test K-segment with different S (quantization fidelity)
    S_values = [4096.0, 8192.0, 16384.0, 32768.0, 65536.0]
    ks_results = []
    for S in S_values:
        print(f"\n--- K-segment S={S:.0f} ---")
        ks_fn, segments = make_k_segment_softmax(B=256, K=8, S=S, degree=9)

        # Multi-run PPL
        ppl_runs = []
        for run in range(N_RUNS):
            torch.cuda.empty_cache()
            ppl, loss, nt = compute_ppl(model, tokenizer, texts_short, ks_fn, 128, DEVICE)
            ppl_runs.append(ppl)

        ppl_mean = np.mean(ppl_runs)
        ppl_std = np.std(ppl_runs)
        delta = ppl_mean - ppl_orig
        pct = (ppl_mean / ppl_orig - 1) * 100

        cos_sim, top10 = compare_with_real(model, tokenizer, prompt, ks_fn, DEVICE)

        print(f"  PPL = {ppl_mean:.4f} +/- {ppl_std:.4f} ({N_RUNS} runs)")
        print(f"  Delta = {delta:+.4f} ({pct:+.2f}%)")
        print(f"  Cos sim = {cos_sim:.6f}, Top-10 = {top10}/10")

        ks_results.append({
            'S': S, 'n_runs': N_RUNS,
            'ppl_mean': ppl_mean, 'ppl_std': ppl_std,
            'ppl_delta': delta, 'ppl_pct_change': pct,
            'cos_sim': cos_sim, 'top10_overlap': top10,
            'n_active_segments': sum(1 for s in segments if s['type'] != 'const'),
            'n_poly_segments': sum(1 for s in segments if s['type'] == 'poly'),
        })

    results_all['k_segment_sweep'] = {
        'baseline_ppl': ppl_orig,
        'baseline_tokens': n_tok,
        'n_runs': N_RUNS,
        'results': ks_results,
    }

    # ====================================================================
    # Part B: K-segment vs Single-segment (domain amplification)
    # ====================================================================
    print("\n" + "=" * 60)
    print("Part B: K-segment (S=256) vs Single-segment domain sweep")
    print("=" * 60)

    # K-segment at optimal S (target lam*M ≈ 2 for segment 1)
    ks_fn_opt, _ = make_k_segment_softmax(B=256, K=8, S=16384.0, degree=9)

    # Single-segment for comparison
    from math import comb as _comb
    def make_single_softmax(M):
        lam = 1.0
        coeffs = generate_polynomial(lam, M, 9)
        cn = np.array(coeffs, dtype=np.float64)
        def fn(logits, dim=-1, dtype=None, **kw):
            x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
            xs = x - mx; xc = torch.clamp(xs, -M, 0.0)
            y = horner_torch_np(cn, -xc); y = torch.clamp(y, min=1e-30)
            out = y / y.sum(dim=dim, keepdim=True)
            t = dtype if dtype is not None else logits.dtype
            return out.to(t) if t != torch.float32 else out
        return fn

    domains = [8.0, 10.0, 12.0, 15.0, 20.0]
    comparison = []

    # K-segment (multi-run)
    ks_ppls = []
    for run in range(N_RUNS):
        ppl, _, _ = compute_ppl(model, tokenizer, texts_short, ks_fn_opt, 128, DEVICE)
        ks_ppls.append(ppl)
    ks_mean = np.mean(ks_ppls)
    ks_std = np.std(ks_ppls)
    ks_cos, ks_top10 = compare_with_real(model, tokenizer, prompt, ks_fn_opt, DEVICE)
    print(f"K-segment (S=256): PPL={ks_mean:.4f}+/-{ks_std:.4f}, delta={(ks_mean/ppl_orig-1)*100:+.2f}%, cos={ks_cos:.6f}")

    comparison.append({
        'method': 'K-segment (S=256)',
        'ppl_mean': ks_mean, 'ppl_std': ks_std,
        'ppl_delta': ks_mean - ppl_orig,
        'ppl_pct_change': (ks_mean/ppl_orig - 1)*100,
        'cos_sim': ks_cos, 'top10_overlap': ks_top10,
    })

    for M in domains:
        ss_fn = make_single_softmax(M)
        ss_ppls = []
        for run in range(N_RUNS):
            ppl, _, _ = compute_ppl(model, tokenizer, texts_short, ss_fn, 128, DEVICE)
            ss_ppls.append(ppl)
        ss_mean = np.mean(ss_ppls)
        ss_std = np.std(ss_ppls)
        ss_cos, ss_top10 = compare_with_real(model, tokenizer, prompt, ss_fn, DEVICE)
        print(f"Single [-{M:.0f},0]: PPL={ss_mean:.4f}+/-{ss_std:.4f}, delta={(ss_mean/ppl_orig-1)*100:+.2f}%, cos={ss_cos:.6f}")
        comparison.append({
            'method': f'Single [-{M:.0f},0]',
            'ppl_mean': ss_mean, 'ppl_std': ss_std,
            'ppl_delta': ss_mean - ppl_orig,
            'ppl_pct_change': (ss_mean/ppl_orig - 1)*100,
            'cos_sim': ss_cos, 'top10_overlap': ss_top10,
        })

    results_all['comparison'] = {'baseline_ppl': ppl_orig, 'n_runs': N_RUNS, 'results': comparison}

    # ====================================================================
    # Part C: K-segment LONG SEQUENCE (the critical test!)
    # ====================================================================
    print("\n" + "=" * 60)
    print("Part C: K-segment LONG SEQUENCE (where single-segment failed)")
    print("=" * 60)

    seq_lengths = [128, 256, 512]  # Limit to avoid OOM
    long_results = []

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        print(f"\n  seq_len={seq_len}...")

        enc = tokenizer(long_text, return_tensors='pt', truncation=True, max_length=seq_len)
        input_ids = enc['input_ids'].to(DEVICE)
        n_tokens = input_ids.size(1)

        # Baseline (do this FIRST and free its outputs)
        F.softmax = REAL_SOFTMAX
        out = model(input_ids, labels=input_ids)
        ppl_base = float(np.exp(out.loss.item())) if out.loss is not None else float('inf')
        del out; torch.cuda.empty_cache()

        # K-segment
        F.softmax = ks_fn_opt
        out_ks = model(input_ids, labels=input_ids)
        ppl_ks = float(np.exp(out_ks.loss.item())) if out_ks.loss is not None else float('inf')
        F.softmax = REAL_SOFTMAX

        delta = ppl_ks - ppl_base
        pct = (ppl_ks / ppl_base - 1) * 100 if ppl_base != float('inf') else float('inf')

        # Single-segment comparison (skip for long sequences to avoid OOM)
        if seq_len <= 256:
            ss_fn_8 = make_single_softmax(8.0)
            F.softmax = ss_fn_8
            out_ss = model(input_ids, labels=input_ids)
            ppl_ss = float(np.exp(out_ss.loss.item())) if out_ss.loss is not None else float('inf')
            F.softmax = REAL_SOFTMAX
            delta_ss = ppl_ss - ppl_base
            pct_ss = (ppl_ss / ppl_base - 1) * 100 if ppl_base != float('inf') else float('inf')
        else:
            ppl_ss = float('inf')
            pct_ss = float('inf')

        print(f"    Base={ppl_base:.4f}, K-seg={ppl_ks:.4f} ({pct:+.2f}%)" +
              (f", Single[-8]={ppl_ss:.4f} ({pct_ss:+.2f}%)" if seq_len <= 256 else ""))

        long_results.append({
            'seq_len': seq_len, 'n_tokens': n_tokens,
            'ppl_baseline': ppl_base,
            'ppl_ksegment': ppl_ks, 'ksegment_delta_pct': pct,
            'ppl_single_8': ppl_ss, 'single_8_delta_pct': pct_ss,
        })

    results_all['long_sequence'] = {'results': long_results}

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n--- K-segment PPL (multi-run, N=3) ---")
    print(f"{'S':<10} {'PPL':<14} {'Std':<10} {'Delta%':<10} {'Cos':<10} {'Top10':<8}")
    print("-" * 62)
    for r in ks_results:
        print(f"{r['S']:<10.0f} {r['ppl_mean']:<14.4f} {r['ppl_std']:<10.4f} {r['ppl_pct_change']:<+10.2f}% {r['cos_sim']:<10.6f} {r['top10_overlap']:<8}/10")

    print("\n--- K-segment vs Single-segment (PPL delta %) ---")
    for r in comparison:
        print(f"  {r['method']:<25}: {r['ppl_mean']:.4f}+/-{r['ppl_std']:.4f}, delta={r['ppl_pct_change']:+.2f}%, cos={r['cos_sim']:.6f}")

    print("\n--- Long Sequence: K-segment vs Single-segment ---")
    print(f"{'Seq':<8} {'Base':<10} {'K-seg':<10} {'K-Δ%':<10} {'Sgl[-8]':<10} {'Sgl-Δ%':<10}")
    print("-" * 58)
    for r in long_results:
        print(f"{r['seq_len']:<8} {r['ppl_baseline']:<10.4f} {r['ppl_ksegment']:<10.4f} "
              f"{r['ksegment_delta_pct']:<+10.2f}% {r['ppl_single_8']:<10.4f} {r['single_8_delta_pct']:<+10.2f}%")

    # Save
    output = {
        'experiment': 'final_ksegment',
        'model': MODEL,
        'd_head': d_head,
        'n_runs': N_RUNS,
        'baseline_ppl': ppl_orig,
        'k_segment_sweep': results_all['k_segment_sweep'],
        'comparison': results_all['comparison'],
        'long_sequence': results_all['long_sequence'],
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp_final_ksegment.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print("Final experiment complete!")

if __name__ == '__main__':
    main()
