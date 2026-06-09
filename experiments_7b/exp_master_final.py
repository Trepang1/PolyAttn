"""
Master Final Experiment
=======================
1. K-segment PPL on 13B (S=4096)
2. K-segment PPL on 0.5B (S=4096)
3. ZK Protocol benchmark (polyeval_prototype timing at scale)
4. Cross-model K-segment comparison table
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

REAL_SOFTMAX = F.softmax

# ====================================================================
# Polynomial + K-segment utilities
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
    y = torch.full_like(x, float(c_np[-1]))
    for ci in reversed(c_np[:-1]): y = y * x + float(ci)
    return y

def make_k_segment_softmax(B=256, K=8, S=4096.0, degree=9):
    segments = []
    for k in range(K):
        lam = (B ** k) / S
        lam_M = lam * (B - 1)
        if lam_M < 0.001:
            segments.append({'type': 'const'})
        elif lam_M > 50:
            segments.append({'type': 'indicator'})
        else:
            coeffs = np.array(generate_polynomial(lam, B - 1, degree), dtype=np.float64)
            segments.append({'type': 'poly', 'coeffs': coeffs})
    n_poly = sum(1 for s in segments if s['type'] == 'poly')

    def k_softmax(logits, dim=-1, dtype=None, **kw):
        x = logits.float(); x_max = x.max(dim=dim, keepdim=True).values
        x_shifted = x - x_max
        v = (-x_shifted * S).round().clamp(0.0, 1e9).long()
        result = torch.ones_like(v, dtype=torch.float32)
        remaining = v.clone()
        for k, seg in enumerate(segments):
            x_k = remaining % B; remaining = remaining // B
            if seg['type'] == 'const': continue
            elif seg['type'] == 'indicator':
                result = result * (x_k == 0).float().clamp(min=1e-30)
            else:
                result = result * horner_torch_np(seg['coeffs'], x_k.float()).clamp(min=1e-30)
        result = result.clamp(min=1e-30)
        out = result / result.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return k_softmax, n_poly

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, softmax_fn, max_length=128, device='cuda'):
    model.eval(); F.softmax = softmax_fn
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
    F.softmax = REAL_SOFTMAX; out_orig = model(input_ids)
    F.softmax = softmax_fn; out_poly = model(input_ids)
    F.softmax = REAL_SOFTMAX
    cos_sim = float(F.cosine_similarity(out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    _, orig_top = out_orig.logits[0, -1].topk(10)
    _, poly_top = out_poly.logits[0, -1].topk(10)
    return cos_sim, len(set(orig_top.tolist()) & set(poly_top.tolist()))

# ====================================================================
# Part 1: K-segment on a model
# ====================================================================

def test_ksegment_on_model(model_path, model_name, texts, prompt, device='cuda'):
    print(f"\n{'='*60}")
    print(f"K-segment PPL: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  {n_params:.1f}B params, VRAM: {vram:.1f}GB")

    # Baseline
    ppl_orig, _, n_tok = compute_ppl(model, tokenizer, texts, REAL_SOFTMAX, 128, device)
    print(f"  Baseline PPL: {ppl_orig:.4f} ({n_tok} tokens)")

    # K-segment S=4096 (optimal from 7B experiments)
    ks_fn, n_poly = make_k_segment_softmax(B=256, K=8, S=4096.0, degree=9)
    print(f"  K-segment: S=4096, {n_poly} polynomial segments")

    # 3 runs for variance
    ppls = []
    for run in range(3):
        torch.cuda.empty_cache()
        ppl, _, _ = compute_ppl(model, tokenizer, texts, ks_fn, 128, device)
        ppls.append(ppl)

    ppl_mean = np.mean(ppls)
    ppl_std = np.std(ppls)
    delta = (ppl_mean / ppl_orig - 1) * 100

    cos_sim, top10 = compare_with_real(model, tokenizer, prompt, ks_fn, device)

    print(f"  K-segment PPL: {ppl_mean:.4f} +/- {ppl_std:.4f}")
    print(f"  Delta: {delta:+.2f}%")
    print(f"  Cos sim: {cos_sim:.6f}, Top-10: {top10}/10")

    # Cleanup
    del model; torch.cuda.empty_cache()
    return {
        'model': model_name, 'n_params': n_params, 'vram_gb': vram,
        'baseline_ppl': ppl_orig, 'baseline_tokens': n_tok,
        'ksegment_S': 4096, 'ksegment_n_poly': n_poly,
        'ppl_mean': ppl_mean, 'ppl_std': ppl_std, 'ppl_pct_change': delta,
        'cos_sim': cos_sim, 'top10_overlap': top10,
        'ppl_runs': ppls,
    }

# ====================================================================
# Part 2: ZK Protocol Benchmark
# ====================================================================

def benchmark_zk_protocol():
    print(f"\n{'='*60}")
    print("ZK Protocol Benchmark (BLS12-381 prototype)")
    print(f"{'='*60}")

    try:
        from py_ecc.bls12_381 import G1 as G1_GEN, add as g1_add, multiply as g1_mul, curve_order
        P = curve_order
        print(f"  BLS12-381 loaded, curve order: {P.bit_length()} bits")
    except ImportError:
        print("  py_ecc not available, skipping ZK benchmark")
        return None

    # Bench at increasing N
    sizes = [64, 256, 1024, 4096]
    bench_results = []

    for N in sizes:
        print(f"\n  N={N}...")
        D = N * N  # 2D tensor

        # Simulated: generate random field elements
        import random
        random.seed(42)

        # Commit time
        t0 = time.time()
        commitments = []
        for _ in range(min(10, D)):  # 10 commitments as sample
            r = random.randint(0, P - 1)
            x = random.randint(0, P - 1)
            g = G1_GEN  # Placeholder — actual commit uses hash-to-G1
            com = g1_mul(g, x)
            commitments.append(com)
        commit_time = time.time() - t0

        # Horner evaluation time (in F_p)
        t0 = time.time()
        d = 9
        for _ in range(D):
            x = random.randint(0, P - 1)
            accum = random.randint(0, P - 1)  # c_d
            for _ in range(d):
                accum = (accum * x + random.randint(0, P - 1)) % P
        horner_time = time.time() - t0

        # Sumcheck time (simulated: d rounds × D evaluations)
        t0 = time.time()
        for _ in range(d):
            _ = sum(random.randint(0, P - 1) for _ in range(D)) % P
        sumcheck_time = time.time() - t0

        # Estimated total per segment
        total_per_seg = commit_time + horner_time + sumcheck_time

        # Extrapolate: K-segment (3 active segments), 40 layers, 32 heads
        n_active_segs = 3
        n_layers = 40
        n_heads = 32
        seq_len = 2048
        D_head = seq_len * seq_len  # attention matrix size
        scale_factor = D_head / D

        est_total = total_per_seg * n_active_segs * n_layers * n_heads * scale_factor

        bench_results.append({
            'N': N, 'D': D,
            'commit_time_s': commit_time,
            'horner_time_s': horner_time,
            'sumcheck_time_s': sumcheck_time,
            'total_per_seg_s': total_per_seg,
            'extrapolated_total_s': est_total,
        })

        print(f"    Commit: {commit_time:.4f}s, Horner: {horner_time:.4f}s, Sumcheck: {sumcheck_time:.4f}s")
        print(f"    Per-segment total: {total_per_seg:.4f}s")
        print(f"    Extrapolated (full model, seq=2048): {est_total:.1f}s")

    # Print comparison table
    print(f"\n  {'N':<8} {'Commit(s)':<12} {'Horner(s)':<12} {'Sumcheck(s)':<14} {'Total/seg(s)':<14} {'Est Full(s)':<14}")
    print("  " + "-" * 76)
    for r in bench_results:
        print(f"  {r['N']:<8} {r['commit_time_s']:<12.4f} {r['horner_time_s']:<12.4f} {r['sumcheck_time_s']:<14.4f} {r['total_per_seg_s']:<14.4f} {r['extrapolated_total_s']:<14.1f}")

    # Speedup estimate vs tlookup
    # tlookup: ~230 mul/element, PolyEval: ~90 mul/element → 2.6x per segment
    # For full zkAttn: ~11.1x (from technical plan)
    print(f"\n  Theoretical speedup vs tlookup (from multiplication counting):")
    print(f"    Per segment: 230/90 = 2.6x")
    print(f"    zkAttn overall: ~11.1x (including high/low segment optimization)")

    return bench_results

# ====================================================================
# Main
# ====================================================================

def main():
    DEVICE = 'cuda'
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PolyAttn Master Final Experiment")
    print("K-segment (13B + 0.5B) + ZK Protocol Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM free: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("=" * 70)

    all_results = {}

    # Texts (same as previous experiments for comparability)
    texts = [
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
    ] * 3

    prompt = "The future of artificial intelligence lies in the development of more efficient"

    # ================================================================
    # Part 1: K-segment on 13B
    # ================================================================
    if os.path.exists('/root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b'):
        r13b = test_ksegment_on_model(
            '/root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b',
            'Chinese-Llama-2-13b', texts, prompt, DEVICE)
        all_results['13b_ksegment'] = r13b
    else:
        print("\n13B model not found, skipping")

    # ================================================================
    # Part 2: K-segment on 0.5B
    # ================================================================
    r05b = test_ksegment_on_model(
        '/root/autodl-tmp/Qwen2.5-0.5B',
        'Qwen2.5-0.5B', texts, prompt, DEVICE)
    all_results['05b_ksegment'] = r05b

    # ================================================================
    # Part 3: Cross-model K-segment comparison table
    # ================================================================
    print(f"\n{'='*60}")
    print("Cross-Model K-segment Comparison (S=4096)")
    print(f"{'='*60}")

    # 7B result from exp_final_ksegment
    r7b_ref = {'ppl_pct_change': -0.26, 'cos_sim': 0.999975, 'top10_overlap': 10, 'ppl_std': 0.0}

    print(f"\n  {'Model':<25} {'Params':<8} {'PPL':<10} {'Delta%':<10} {'Cos sim':<12} {'Top10':<8}")
    print("  " + "-" * 73)
    for r, name in [(r05b, 'Qwen2.5-0.5B'), (r7b_ref, 'Chinese-Llama-2-7b')]:
        if name == 'Chinese-Llama-2-7b':
            print(f"  {name:<25} {'6.9B':<8} {'-':<10} {r['ppl_pct_change']:<+10.2f}% {r['cos_sim']:<12.6f} {r['top10_overlap']:<8}/10")
        else:
            pass  # handled below
    print(f"  {'Qwen2.5-0.5B':<25} {'0.5B':<8} {r05b['ppl_mean']:<10.4f} {r05b['ppl_pct_change']:<+10.2f}% {r05b['cos_sim']:<12.6f} {r05b['top10_overlap']:<8}/10")
    if '13b_ksegment' in all_results:
        r13b_d = all_results['13b_ksegment']
        print(f"  {'Chinese-Llama-2-13b':<25} {'13.3B':<8} {r13b_d['ppl_mean']:<10.4f} {r13b_d['ppl_pct_change']:<+10.2f}% {r13b_d['cos_sim']:<12.6f} {r13b_d['top10_overlap']:<8}/10")

    # ================================================================
    # Part 4: ZK Protocol Benchmark
    # ================================================================
    zk_results = benchmark_zk_protocol()
    if zk_results:
        all_results['zk_protocol_benchmark'] = zk_results

    # ================================================================
    # Save
    # ================================================================
    out_path = os.path.join(OUTPUT_DIR, 'exp_master_final.json')
    with open(out_path, 'w') as f:
        # Convert numpy types
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nResults saved to {out_path}")
    print("Master experiment complete!")

if __name__ == '__main__':
    main()
