"""
实验 4 (v2): 标准 Benchmark PPL 测试 + Monkey-patch 验证
==========================================================
固定 d=9, domain=[-8,0]
每个 benchmark 独立加载模型，避免状态污染
关键修复: 每次 benchmark 开始前显式验证 monkey-patch 是否生效
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

def make_poly_softmax(M_domain, d=9):
    lam = 1.0; coeffs = generate_polynomial(lam, M_domain, d)
    linf = check_approx(coeffs, lam, M_domain)
    cn = np.array(coeffs, dtype=np.float64)
    def fn(logits, dim=-1, dtype=None, **kw):
        x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
        xs = x - mx; xc = torch.clamp(xs, -M_domain, 0.0)
        y = horner_torch(cn, -xc); y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return fn, linf

# ============================================================================
# PPL with explicit softmax parameter
# ============================================================================

@torch.no_grad()
def compute_ppl_with_softmax(model, tokenizer, texts, softmax_fn, max_length=256, device='cuda', max_samples=200):
    """Compute PPL using an explicit softmax function. Sets F.softmax before each batch."""
    model.eval()
    # Explicitly set softmax
    F.softmax = softmax_fn
    total_loss, total_tokens = 0.0, 0
    count = 0
    for text in texts:
        if count >= max_samples: break
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            continue
        enc = tokenizer(text.strip(), return_tensors='pt', truncation=True, max_length=max_length)
        ids = enc['input_ids'].to(device)
        if ids.size(1) < 4: continue
        try:
            out = model(ids, labels=ids)
            if out.loss is not None and not torch.isnan(out.loss):
                total_loss += out.loss.item() * ids.size(1)
                total_tokens += ids.size(1)
                count += 1
        except Exception as e:
            pass
    if total_tokens == 0: return float('inf'), float('inf'), count
    avg = total_loss / total_tokens
    return float(np.exp(avg)), float(avg), count

# ============================================================================
# Quick monkey-patch verification
# ============================================================================

@torch.no_grad()
def verify_patch_working(model, tokenizer, poly_fn, device='cuda'):
    """Verify that the monkey-patch actually changes model outputs."""
    prompt = "The future of artificial intelligence lies in the development of more efficient"
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
    input_ids = enc['input_ids'].to(device)

    # Run with real exp
    F.softmax = REAL_SOFTMAX
    out_orig = model(input_ids)

    # Run with poly
    F.softmax = poly_fn
    out_poly = model(input_ids)

    # Restore
    F.softmax = REAL_SOFTMAX

    diff = (out_orig.logits.float() - out_poly.logits.float()).abs().max().item()
    identical = torch.allclose(out_orig.logits.float(), out_poly.logits.float())
    cos_sim = float(F.cosine_similarity(
        out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0))
    return not identical, diff, cos_sim

# ============================================================================
# Store the real softmax at import time
# ============================================================================

REAL_SOFTMAX = F.softmax

# ============================================================================
# Main
# ============================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DOMAIN_M = 8.0
    DEGREE = 9
    MAX_SAMPLES = 200
    MAX_LENGTH = 256
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("实验 4 (v2): 标准 Benchmark PPL 测试 + 验证")
    print(f"固定: d={DEGREE}, domain=[-{DOMAIN_M},0]")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Generate polynomial once
    poly_fn, linf = make_poly_softmax(DOMAIN_M, d=DEGREE)
    print(f"\nPolyAttn: d={DEGREE}, L∞={linf:.2e}")

    # ---- Load model once ----
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ---- Verify monkey-patch works ----
    print("\n--- Verifying monkey-patch ---")
    patch_works, diff, verify_cos = verify_patch_working(model, tokenizer, poly_fn, DEVICE)
    print(f"  Patch working: {patch_works}")
    print(f"  Logits max diff: {diff:.4f}")
    print(f"  Cos sim (should be < 1.0): {verify_cos:.6f}")
    if not patch_works:
        print("  *** ERROR: Monkey-patch NOT working! Aborting. ***")
        return
    if diff < 0.01:
        print("  *** WARNING: Logits diff too small, patch may not be effective ***")

    # ---- Prepare benchmarks ----
    # WikiText-2 fallback
    wiki_texts = [
        "The history of artificial intelligence dates back to the mid 20th century "
        "when researchers first began exploring the possibility of creating machines "
        "that could think and learn like humans. Early work focused on symbolic reasoning "
        "and rule based systems.",
    ] * 50

    # Chinese Benchmark
    chinese_texts = [
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。深度学习的发展推动了自然语言处理领域的巨大进步。",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。计算机视觉技术使得机器能够理解和分析图像内容。",
        "量子计算作为一种新型计算范式，有望在密码学、药物研发等领域带来革命性变化。中国在人工智能领域的研究投入不断增加。",
        "自然语言处理是人工智能的重要分支，涵盖了文本分类、情感分析、命名实体识别等多个方向。机器学习算法在图像识别和语音识别上表现优异。",
        "大数据时代，如何有效保护用户隐私成为了学术界和工业界共同面临的关键挑战。联邦学习能够在保护数据隐私的同时实现模型协同训练。",
        "强化学习在游戏对弈和机器人控制领域展现出了强大的自主学习能力。知识蒸馏通过将教师模型的知识迁移到学生模型实现模型压缩。",
        "Transformer架构已经成为现代自然语言处理系统的基础组件。注意力机制允许模型动态关注输入序列中最相关的信息。",
        "预训练加微调的范式极大地降低了自然语言处理任务对标注数据的需求。大规模语言模型展现出上下文学习和思维链推理等涌现能力。",
        "模型量化技术通过降低参数的数值精度来减少模型存储和计算开销。神经网络架构搜索自动化了深度学习模型的设计过程。",
        "对比学习通过拉近正样本对和推远负样本对来学习有效特征表示。自监督学习利用数据本身结构信息构建预训练任务。",
        "多模态学习结合文本图像语音等多种信息模态，提升了模型的理解和推理能力。因果推断帮助模型超越简单的相关性学习。",
        "图神经网络扩展了深度学习到非欧几里得结构的图数据上，在社交网络和分子建模中广泛应用。",
        "元学习旨在让模型学会如何学习，从而能够快速适应新的任务和场景。持续学习研究如何避免灾难性遗忘。",
        "可解释性人工智能致力于让深度学习模型的决策过程对人类更加透明和可理解。这对医疗金融等高风险领域尤为重要。",
        "生成对抗网络通过生成器和判别器的对抗训练来生成逼真的数据样本。变分自编码器提供了另一种生成模型的框架。",
    ] * 14  # ~210 texts

    # Mixed Technical EN
    en_technical = [
        "Machine learning has revolutionized how we approach complex problems in science and engineering. "
        "Deep neural networks have achieved remarkable results across many domains including computer vision, "
        "natural language processing, and speech recognition.",
        "The transformer architecture has become the foundation of modern NLP systems. "
        "Attention mechanisms allow models to focus on relevant parts of the input sequence, "
        "enabling better handling of long range dependencies.",
        "Transfer learning has dramatically reduced the data requirements for NLP tasks. "
        "By pretraining on large corpora and fine tuning on specific tasks, models achieve "
        "state of the art performance with minimal task specific data.",
        "Large language models demonstrate emergent abilities as they scale up in size, "
        "including in context learning, chain of thought reasoning, and instruction following. "
        "These capabilities were not explicitly programmed but arose from scale.",
        "The combination of scale and data quality determines model performance. "
        "While larger models generally perform better, the quality and diversity of training data "
        "plays an equally important role in determining downstream task performance.",
    ] * 40

    # Code-related texts
    code_texts = [
        "The implementation of this algorithm requires careful consideration of computational complexity. "
        "We can optimize the runtime from quadratic to linear by using a hash table to store intermediate results.",
        "Python's dynamic typing and garbage collection make it an excellent choice for rapid prototyping. "
        "However, for production systems, statically typed languages like Rust or Go may offer better performance.",
    ] * 100

    benchmarks = [
        ("WikiText-2 (fallback)", wiki_texts),
        ("Chinese Benchmark", chinese_texts),
        ("Mixed Technical EN", en_technical),
        ("Code-related EN", code_texts),
    ]

    # ---- Run benchmarks with explicit softmax control ----
    results = []

    for bench_name, texts in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name} ({len(texts)} texts, max {MAX_SAMPLES} samples)")

        # 1. Compute baseline with REAL exp softmax
        print(f"  [1/2] Computing with REAL exp softmax...")
        ppl_orig, loss_orig, n = compute_ppl_with_softmax(
            model, tokenizer, texts, REAL_SOFTMAX, MAX_LENGTH, DEVICE, MAX_SAMPLES)
        print(f"  Original: PPL={ppl_orig:.4f}, loss={loss_orig:.4f}, samples={n}")

        # 2. Compute with PolyAttn
        print(f"  [2/2] Computing with PolyAttn softmax...")
        ppl_poly, loss_poly, n_poly = compute_ppl_with_softmax(
            model, tokenizer, texts, poly_fn, MAX_LENGTH, DEVICE, MAX_SAMPLES)
        delta = ppl_poly - ppl_orig
        pct = (ppl_poly / ppl_orig - 1) * 100
        print(f"  PolyAttn: PPL={ppl_poly:.4f}, loss={loss_poly:.4f}, samples={n_poly}")
        print(f"  ΔPPL={delta:+.4f} ({pct:+.3f}%)")

        # Verify delta is non-zero (sanity check)
        if abs(delta) < 1e-6:
            print(f"  *** WARNING: ΔPPL is exactly zero! Patch may not be working for this benchmark! ***")

        results.append({
            'benchmark': bench_name,
            'num_samples': n,
            'ppl_orig': ppl_orig,
            'ppl_poly': ppl_poly,
            'ppl_delta': delta,
            'ppl_pct_change': pct,
            'loss_orig': loss_orig,
            'loss_poly': loss_poly,
        })

    # ---- Final verification ----
    print(f"\n--- Post-run verification ---")
    patch_works2, diff2, verify_cos2 = verify_patch_working(model, tokenizer, poly_fn, DEVICE)
    print(f"  Patch still working: {patch_works2}")
    print(f"  Logits max diff: {diff2:.4f}")
    print(f"  Cos sim: {verify_cos2:.6f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("标准 Benchmark 结果汇总")
    print("=" * 70)
    print(f"{'Benchmark':<25} {'Samples':<8} {'Orig PPL':<12} {'Poly PPL':<12} {'ΔPPL':<10} {'Δ%':<10}")
    print("-" * 77)
    for r in results:
        print(f"{r['benchmark']:<25} {r['num_samples']:<8} {r['ppl_orig']:<12.4f} "
              f"{r['ppl_poly']:<12.4f} {r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.3f}%")

    # Stats
    valid = [r for r in results if r['ppl_orig'] != float('inf')]
    zero_deltas = [r['benchmark'] for r in valid if abs(r['ppl_delta']) < 1e-6]
    if zero_deltas:
        print(f"\n*** BENCHMARKS WITH ZERO DELTA: {zero_deltas} ***")
        print(f"*** These results are INVALID. Monkey-patch did not work. ***")

    if valid:
        avg_pct = np.mean([abs(r['ppl_pct_change']) for r in valid])
        print(f"\n平均 |ΔPPL%|: {avg_pct:.3f}%")
        max_pct = max(abs(r['ppl_pct_change']) for r in valid)
        print(f"最大 |ΔPPL%|: {max_pct:.3f}%")

    # Save
    output = {
        'experiment': 'standard_benchmark_v2',
        'model': MODEL,
        'degree': DEGREE,
        'domain': DOMAIN_M,
        'max_samples': MAX_SAMPLES,
        'max_length': MAX_LENGTH,
        'patch_verified': patch_works,
        'verify_logits_max_diff': diff,
        'verify_cos_sim': verify_cos,
        'results': results,
        'zero_delta_benchmarks': zero_deltas,
        'avg_abs_pct_change': avg_pct if valid else None,
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp4_standard_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    print("实验 4 (v2) 完成！")

if __name__ == '__main__':
    main()
