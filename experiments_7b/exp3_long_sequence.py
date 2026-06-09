"""
实验 3: 长序列 PPL 测试
=======================
固定 d=9, domain=[-8,0], 扫描 seq_len ∈ {128, 256, 512, 1024, 2048}
目标: 验证长序列下误差不累积
"""
import torch, torch.nn.functional as F, numpy as np, time, json, os, sys
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Polynomial utilities (same as exp1)
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
    return float(np.max(np.abs(np.exp(-lam * grid) - np.polyval(coeffs[::-1], grid))))

def horner_torch(c, x):
    y = torch.full_like(x, c[-1])
    for ci in reversed(c[:-1]): y = y * x + ci
    return y

def make_poly_softmax(M_domain, d=9):
    lam_val = 1.0; coeffs = generate_polynomial(lam_val, M_domain, d)
    linf, cn = check_approx(coeffs, lam_val, M_domain), np.array(coeffs, dtype=np.float64)
    def fn(logits, dim=-1, dtype=None, **kw):
        x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
        xs = x - mx; xc = torch.clamp(xs, -M_domain, 0.0)
        y = horner_torch(cn, -xc); y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        t = dtype if dtype is not None else logits.dtype
        return out.to(t) if t != torch.float32 else out
    return fn, linf

# ============================================================================
# Long text PPL (uses a single long passage, not short texts)
# ============================================================================

@torch.no_grad()
def compute_ppl_long(model, tokenizer, text, max_length, device='cuda'):
    """Compute PPL on one long text, sliding window style."""
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = enc['input_ids'].to(device)
    if input_ids.size(1) < 4:
        return float('inf'), float('inf'), 0
    try:
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        if loss is not None and not torch.isnan(loss):
            n_tokens = input_ids.size(1)
            return float(np.exp(loss.item())), float(loss.item()), n_tokens
    except Exception as e:
        print(f"    Error: {e}")
    return float('inf'), float('inf'), 0

# ============================================================================
# Main
# ============================================================================

def main():
    MODEL = '/root/autodl-tmp/chinese-llama-2-7b'
    DEVICE = 'cuda'
    DOMAIN_M = 8.0
    DEGREE = 9
    SEQ_LENGTHS = [128, 256, 512, 1024, 2048]
    OUTPUT_DIR = '/root/autodl-tmp/experiments_7b_results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("实验 3: 长序列 PPL 测试")
    print(f"固定: d={DEGREE}, domain=[-{DOMAIN_M},0]")
    print(f"序列长度: {SEQ_LENGTHS}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
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

    # Long Chinese passage for testing
    # We'll use a concatenated passage from multiple texts
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
        "大规模语言模型随着参数量的增加展现出令人惊讶的涌现能力，如上下文学习和思维链推理。"
        "数据质量和规模的结合是决定模型最终性能的关键因素，两者缺一不可。"
        "联邦学习作为一种分布式机器学习范式，能够在保护数据隐私的同时实现模型协同训练。"
        "差分隐私技术通过添加精心设计的噪声来保护个体数据在统计查询中的隐私安全。"
        "同态加密允许在加密数据上直接进行计算，为隐私保护机器学习提供了理论支持。"
        "零知识证明技术使得一方可以向另一方证明某个陈述为真而不泄露任何额外信息。"
        "知识蒸馏通过将大型教师模型的知识迁移到小型学生模型，实现模型压缩和加速。"
        "模型量化技术通过降低参数的数值精度来减少模型的存储和计算开销。"
        "神经网络架构搜索自动化了深度学习模型的设计过程，减少了人工调参的工作量。"
        "对比学习通过拉近正样本对、推远负样本对来学习有效的特征表示。"
        "自监督学习利用数据本身的结构信息来构建预训练任务，无需人工标注。"
        "多模态学习结合文本、图像、语音等多种信息模态，提升了模型的理解和推理能力。"
        "因果推断技术帮助机器学习模型超越简单的相关性学习，理解变量之间的因果关系。"
        "图神经网络扩展了深度学习到非欧几里得结构的图数据上，在社交网络和分子建模中广泛应用。"
        "元学习旨在让模型学会如何学习，从而能够快速适应新的任务和场景。"
        "持续学习研究如何让模型在顺序学习新任务时不会灾难性地遗忘旧知识。"
        "可解释性人工智能致力于让深度学习模型的决策过程对人类更加透明和可理解。"
    )

    # Repeat to fill max sequence length
    long_text = (long_text_base + " ") * 30  # ~15K chars

    # Generate polynomial once
    poly_fn, linf = make_poly_softmax(DOMAIN_M, d=DEGREE)
    print(f"\nPolyAttn: d={DEGREE}, L∞={linf:.2e}")

    orig_softmax = F.softmax
    results = []

    for seq_len in SEQ_LENGTHS:
        print(f"\n--- seq_len={seq_len} ---")
        torch.cuda.empty_cache()
        t0_vram = torch.cuda.memory_allocated() / 1e9

        # Original
        F.softmax = orig_softmax
        try:
            t0 = time.time()
            ppl_orig, loss_orig, n_tok_orig = compute_ppl_long(model, tokenizer, long_text, seq_len, DEVICE)
            dt_orig = time.time() - t0
            print(f"  Original: PPL={ppl_orig:.4f}, loss={loss_orig:.4f}, "
                  f"tokens={n_tok_orig}, time={dt_orig:.1f}s")
        except torch.cuda.OutOfMemoryError:
            print(f"  Original: OOM at seq_len={seq_len}!")
            results.append({'seq_len': seq_len, 'status': 'OOM_original', 'ppl_orig': None, 'ppl_poly': None})
            break

        # PolyAttn
        F.softmax = poly_fn
        try:
            t0 = time.time()
            ppl_poly, loss_poly, n_tok_poly = compute_ppl_long(model, tokenizer, long_text, seq_len, DEVICE)
            dt_poly = time.time() - t0
            delta = ppl_poly - ppl_orig
            pct = (ppl_poly / ppl_orig - 1) * 100
            vram_peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  PolyAttn: PPL={ppl_poly:.4f}, loss={loss_poly:.4f}, "
                  f"tokens={n_tok_poly}, time={dt_poly:.1f}s")
            print(f"  ΔPPL={delta:+.4f} ({pct:+.3f}%), peak VRAM={vram_peak:.2f}GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  PolyAttn: OOM at seq_len={seq_len}!")
            results.append({'seq_len': seq_len, 'status': 'OOM_poly', 'ppl_orig': ppl_orig, 'ppl_poly': None})
            break

        results.append({
            'seq_len': seq_len,
            'status': 'OK',
            'ppl_orig': ppl_orig,
            'ppl_poly': ppl_poly,
            'ppl_delta': delta,
            'ppl_pct_change': pct,
            'loss_orig': loss_orig,
            'loss_poly': loss_poly,
            'num_tokens': n_tok_orig,
            'time_orig_s': dt_orig,
            'time_poly_s': dt_poly,
            'peak_vram_gb': vram_peak,
        })

    F.softmax = orig_softmax

    # Summary
    print("\n" + "=" * 70)
    print("长序列 PPL 测试结果汇总")
    print("=" * 70)
    print(f"{'SeqLen':<10} {'Orig PPL':<12} {'Poly PPL':<12} {'ΔPPL':<10} {'Δ%':<10} {'VRAM':<10}")
    print("-" * 64)
    for r in results:
        if r['status'] != 'OK':
            print(f"{r['seq_len']:<10} {'FAILED: '+r['status']}")
        else:
            print(f"{r['seq_len']:<10} {r['ppl_orig']:<12.4f} {r['ppl_poly']:<12.4f} "
                  f"{r['ppl_delta']:<+10.4f} {r['ppl_pct_change']:<+10.3f}% {r['peak_vram_gb']:<10.2f}")

    # Check trend
    ok_results = [r for r in results if r['status'] == 'OK']
    if len(ok_results) >= 2:
        deltas = [r['ppl_delta'] for r in ok_results]
        trend = "increasing" if deltas[-1] > deltas[0] else "stable/decreasing"
        print(f"\nPPL delta trend: {trend} (from seq={ok_results[0]['seq_len']} to seq={ok_results[-1]['seq_len']})")

    # Save
    output = {
        'experiment': 'long_sequence',
        'model': MODEL,
        'degree': DEGREE,
        'domain': DOMAIN_M,
        'results': results,
    }
    out_path = os.path.join(OUTPUT_DIR, 'exp3_long_sequence.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    print("实验 3 完成！")

if __name__ == '__main__':
    main()
