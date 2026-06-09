"""
PolyAttn PPL Test -- Real LLaMA Weights (Fixed)
================================================
Replaces attention Softmax exp with PolyAttn's K-segment polynomial approach.
Uses validated Chebyshev→monomial conversion from polyeval_prototype.py.

The key insight: a single global polynomial for exp(x) on [-15,0] is impractical
at d=9. PolyAttn works because of K-segment decomposition -- each segment
covers a different λ_k range. For the PPL test, we:
1. Replace exp in Softmax with the polynomial APPROPRIATE for each λ_k segment
2. Measure PPL degradation

For a direct Softmax test (not full zkLLM decomposition), we use a single
polynomial on a narrower domain [-8, 0] where most attention scores fall.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time, argparse, os, sys
from math import comb

# ============================================================================
# Validated Chebyshev polynomial generation (from polyeval_prototype.py)
# ============================================================================

def generate_polynomial(lambda_val: float, M: float, d: int) -> list:
    """Monomial coeffs for exp(-λx) on [0,M]. Matches polyeval_prototype.py."""
    n_cheb = max(d * 12, 64)
    k = np.arange(n_cheb)
    t_nodes = np.cos(np.pi * (k + 0.5) / n_cheb)
    x_nodes = (M / 2.0) * (t_nodes + 1.0)
    f_vals = np.exp(-lambda_val * x_nodes)

    # DCT-I -> Chebyshev coefficients
    cheb = np.zeros(n_cheb)
    for k_idx in range(n_cheb):
        cheb[k_idx] = (2.0 / n_cheb) * np.sum(f_vals * np.cos(k_idx * np.pi * (k + 0.5) / n_cheb))
    cheb[0] /= 2.0
    cheb = cheb[:d + 1]

    # T_k(y) monomial coefficients (y in [-1,1])
    T = np.zeros((d + 1, d + 1))
    T[0, 0] = 1.0
    if d >= 1:
        T[1, 1] = 1.0
    for k_idx in range(2, d + 1):
        T[k_idx, 1:] += 2.0 * T[k_idx - 1, :-1]
        T[k_idx, :] -= T[k_idx - 2, :]

    # Convert: sum c_k * T_k(2x/M-1) -> sum a_r * x^r
    mono = np.zeros(d + 1)
    for k_idx in range(d + 1):
        for j in range(k_idx + 1):
            t_kj = T[k_idx, j]
            if abs(t_kj) < 1e-16:
                continue
            for r in range(j + 1):
                term = t_kj * cheb[k_idx] * comb(j, r) * (2.0 / M) ** r * (-1.0) ** (j - r)
                mono[r] += term

    return list(mono)


def check_approximation(coeffs: list, lam: float, M: float) -> tuple:
    """Return (Linf_error, max_rel_error, samples) for exp(-lam*x) on [0,M]."""
    grid = np.linspace(0, M, 5001)
    f_true = np.exp(-lam * grid)
    p_vals = np.polyval(coeffs[::-1], grid)
    abs_err = np.abs(f_true - p_vals)
    rel_err = np.abs(f_true - p_vals) / np.maximum(f_true, 1e-30)
    linf = float(np.max(abs_err))
    max_rel = float(np.max(rel_err[f_true > 1e-15]))
    samples = {
        'x=0': (float(f_true[0]), float(p_vals[0])),
        f'x=M/4={M/4:.1f}': (float(f_true[len(grid)//4]), float(p_vals[len(grid)//4])),
        f'x=M/2={M/2:.1f}': (float(f_true[len(grid)//2]), float(p_vals[len(grid)//2])),
        f'x=M={M}': (float(f_true[-1]), float(p_vals[-1])),
    }
    return linf, max_rel, samples


# ============================================================================
# PolyAttn Softmax
# ============================================================================

def horner_torch(coeffs, x):
    """Horner: coeffs=[c_0,...,c_d], returns p(x)=c_0 + c_1*x + ... + c_d*x^d."""
    y = torch.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        y = y * x + c
    return y


def make_polyattn_softmax(domain_M=8.0, d=9):
    """Create a PolyAttn softmax function using Chebyshev polynomial.

    Approximates exp(x) for x in [-domain_M, 0] via poly for exp(-y), y in [0, domain_M].
    """
    lam = 1.0
    coeffs_float = generate_polynomial(lam, domain_M, d)
    linf, max_rel, samples = check_approximation(coeffs_float, lam, domain_M)

    print(f"  Polynomial: d={d}, domain=[{-domain_M}, 0]")
    print(f"  Linf error: {linf:.2e}, max rel error: {max_rel:.2e}")
    for label, (true, poly) in samples.items():
        rel = abs(true - poly) / max(abs(true), 1e-30) if abs(true) > 1e-15 else float('inf')
        print(f"    {label}: exp={true:.6e}, poly={poly:.6e}, rel_err={rel:.2e}")

    coeffs_np = np.array(coeffs_float, dtype=np.float64)

    def poly_softmax(logits, dim=-1, dtype=None, **kwargs):
        orig_dtype = logits.dtype
        x = logits.float()

        # Subtract max for stability
        x_max = x.max(dim=dim, keepdim=True).values
        x_shifted = x - x_max  # all <= 0

        # Clip to domain
        x_clipped = torch.clamp(x_shifted, -domain_M, 0.0)

        # exp(x) = p(-x) where p fits exp(-y), y in [0, domain_M]
        neg_x = -x_clipped  # >= 0
        y = horner_torch(coeffs_np, neg_x)

        # Safeguard
        y = torch.clamp(y, min=1e-30)

        # Normalize
        out = y / y.sum(dim=dim, keepdim=True)

        # Cast back: if dtype arg given, use it; otherwise keep original
        target_dtype = dtype if dtype is not None else orig_dtype
        if target_dtype != torch.float32:
            out = out.to(target_dtype)
        return out

    return poly_softmax, coeffs_float, linf


# ============================================================================
# PPL measurement
# ============================================================================

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, max_length=256, device='cuda'):
    """Compute perplexity on text list."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = enc['input_ids'].to(device)
        if input_ids.size(1) < 4:
            continue
        try:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        except Exception as e:
            print(f"  Warning: {e}")

    if total_tokens == 0:
        return float('inf'), float('inf')
    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss)), float(avg_loss)


# ============================================================================
# Single-prompt comparison
# ============================================================================

@torch.no_grad()
def compare_outputs(model, tokenizer, prompt, poly_softmax_fn, device='cuda'):
    """Compare original vs PolyAttn outputs."""
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)

    # Original
    out_orig = model(input_ids, output_hidden_states=True)

    # PolyAttn (monkey-patch F.softmax)
    import torch.nn.functional as F_ref
    orig_softmax = F_ref.softmax
    F_ref.softmax = poly_softmax_fn
    try:
        out_poly = model(input_ids, output_hidden_states=True)
    finally:
        F_ref.softmax = orig_softmax

    # Logits comparison
    logits_diff = (out_orig.logits.float() - out_poly.logits.float()).abs()
    cos_sim = float(F_ref.cosine_similarity(
        out_orig.logits.float().view(-1), out_poly.logits.float().view(-1), dim=0
    ))

    # Top-k overlap
    top_k = 10
    _, orig_top = out_orig.logits[0, -1].topk(top_k)
    _, poly_top = out_poly.logits[0, -1].topk(top_k)
    overlap = len(set(orig_top.tolist()) & set(poly_top.tolist()))

    print(f"  Logits max diff:  {logits_diff.max().item():.6e}")
    print(f"  Logits mean diff: {logits_diff.mean().item():.6e}")
    print(f"  Logits cos sim:   {cos_sim:.8f}")
    print(f"  Top-{top_k} overlap:   {overlap}/{top_k}")

    # Per-layer drift
    if out_orig.hidden_states:
        print(f"  Layer drift (every 4th):")
        for i, (h_orig, h_poly) in enumerate(zip(out_orig.hidden_states, out_poly.hidden_states)):
            if h_orig is not None:
                rel = float((h_orig.float() - h_poly.float()).norm() / (h_orig.float().norm() + 1e-10))
                if i % 4 == 0 or i == len(out_orig.hidden_states) - 1:
                    print(f"    L{i:2d}: rel_drift={rel:.6e}")

    logits_max_diff = float(logits_diff.max().item())
    return {
        'logits_max_diff': logits_max_diff,
        'logits_mean_diff': float(logits_diff.mean().item()),
        'logits_cos_sim': cos_sim,
        'topk_overlap': overlap,
    }, logits_max_diff


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/Qwen2.5-0.5B')
    parser.add_argument('--domain', type=float, default=8.0)
    parser.add_argument('--degree', type=int, default=9)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=30)
    parser.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB used / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB total")

    # ---- Generate polynomial ----
    print(f"\n{'='*60}")
    print("PolyAttn Polynomial Generation")
    print(f"{'='*60}")
    poly_softmax_fn, coeffs_float, linf = make_polyattn_softmax(domain_M=args.domain, d=args.degree)

    # ---- Load model ----
    print(f"\n{'='*60}")
    print(f"Loading: {args.model}")
    print(f"{'='*60}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force eager attention (not SDPA) so F.softmax is actually called
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params loaded")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ---- Compare outputs ----
    print(f"\n{'='*60}")
    print("Original vs PolyAttn output comparison")
    print(f"{'='*60}")
    prompt = "The future of artificial intelligence lies in the development of more efficient"
    _, max_logit_diff = compare_outputs(model, tokenizer, prompt, poly_softmax_fn, device)

    if args.compare_only:
        return

    # ---- PPL measurement ----
    print(f"\n{'='*60}")
    print("PPL Measurement (Chinese + English texts)")
    print(f"{'='*60}")

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
    ] * (args.num_samples // 10 + 1)
    texts = texts[:args.num_samples]

    # Original PPL
    import torch.nn.functional as F_ref
    orig_softmax = F_ref.softmax
    print("Computing original PPL...")
    t0 = time.time()
    ppl_orig, loss_orig = compute_ppl(model, tokenizer, texts, args.max_length, device)
    dt_orig = time.time() - t0
    print(f"  Original:  PPL={ppl_orig:.4f}, loss={loss_orig:.4f}, time={dt_orig:.1f}s")

    # PolyAttn PPL
    F_ref.softmax = poly_softmax_fn
    print("Computing PolyAttn PPL...")
    t0 = time.time()
    ppl_poly, loss_poly = compute_ppl(model, tokenizer, texts, args.max_length, device)
    dt_poly = time.time() - t0
    print(f"  PolyAttn:  PPL={ppl_poly:.4f}, loss={loss_poly:.4f}, time={dt_poly:.1f}s")
    F_ref.softmax = orig_softmax

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("PolyAttn PPL Test Summary")
    print(f"{'='*60}")
    print(f"  Model:           {args.model}")
    print(f"  Poly degree:     d={args.degree}")
    print(f"  Domain:          [{-args.domain}, 0]")
    print(f"  Linf error:      {linf:.2e}")
    print(f"  Original PPL:    {ppl_orig:.4f}")
    print(f"  PolyAttn PPL:    {ppl_poly:.4f}")
    print(f"  PPL degradation: {ppl_poly - ppl_orig:.4f} ({(ppl_poly/ppl_orig - 1)*100:.2f}%)")
    print(f"  Logits cos sim:  {max_logit_diff:.6e} (max diff)")
    print(f"  Loss delta:      {loss_poly - loss_orig:.6f}")

    if ppl_poly - ppl_orig < 1.0:
        print(f"\n  *** PPL degradation < 1.0 -- PolyAttn is highly viable! ***")
    elif ppl_poly - ppl_orig < 5.0:
        print(f"\n  ** PPL degradation {ppl_poly-ppl_orig:.1f} -- acceptable, may need tuning **")
    else:
        print(f"\n  * PPL degradation > 5.0 -- needs improvement (try multi-segment approach) *")

    print("Done.")


if __name__ == '__main__':
    main()
