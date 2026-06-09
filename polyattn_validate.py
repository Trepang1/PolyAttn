#!/usr/bin/env python3
"""
PolyAttn Validation Prototype
==============================
验证核心主张: d=7 的 Remez 多项式能否替代 exp 查表用于 Attention Softmax?

四大模块:
  1. Remez 算法实现 — 对 exp(-λx) 在 [0, M] 上生成 d 次 minimax 多项式
  2. 误差分析 — L∞ / L2 误差 vs 多项式次数 d
  3. 模拟 Attention — 随机 Q/K/V 上对比真实 Softmax 和多项式 Softmax
  4. 误差传播 — 多层残差传播后输出漂移估计
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10


# =============================================================================
# Part 1: Remez Exchange Algorithm
# =============================================================================

def remez_approx(f, a, b, d, max_iter=50, tol=1e-12):
    """
    Remez exchange algorithm for minimax polynomial approximation.

    Parameters:
        f:      target function f(x)
        a, b:   interval [a, b]
        d:      polynomial degree
        max_iter: max iterations
        tol:    convergence tolerance

    Returns:
        coeffs: monomial coefficients [c_0, ..., c_d] for sum c_i * x^i
        error:  final L∞ error |f - p|
        points: final alternation points
    """
    # Step 1: initial reference points (Chebyshev nodes)
    k = np.arange(d + 2)
    x = (a + b) / 2 + (b - a) / 2 * np.cos(np.pi * k / (d + 1))
    x = np.sort(x)

    for iteration in range(max_iter):
        # Step 2: solve linear system for coefficients and error E
        # f(x_i) - p(x_i) = (-1)^i * E
        # p(x_i) + (-1)^i * E = f(x_i)
        A = np.zeros((d + 2, d + 2))
        for i in range(d + 2):
            for j in range(d + 1):
                A[i, j] = x[i] ** j
            A[i, d + 1] = (-1) ** i

        b_vec = np.array([f(x_i) for x_i in x])
        sol = np.linalg.solve(A, b_vec)
        coeffs = sol[: d + 1]  # [c_0, c_1, ..., c_d]
        E = sol[d + 1]

        # Step 3: find where |f - p| is maximized
        # Search on a dense grid
        grid = np.linspace(a, b, max(10001, 20 * d))
        p_vals = np.polyval(coeffs[::-1], grid)
        f_vals = f(grid)
        err = f_vals - p_vals
        abs_err = np.abs(err)

        max_idx = np.argmax(abs_err)
        new_x = grid[max_idx]

        # Step 4: replace the nearest reference point with same sign
        new_sign = np.sign(err[max_idx])
        # Find existing point with same sign
        existing_signs = np.array([(-1) ** i for i in range(d + 2)])
        same_sign_idx = np.where(existing_signs == new_sign)[0]

        # Replace the one closest to new_x
        distances = np.abs(x[same_sign_idx] - new_x)
        replace_idx = same_sign_idx[np.argmin(distances)]
        x[replace_idx] = new_x
        x = np.sort(x)

        # Convergence check
        if iteration > 0:
            if abs(abs_E - abs(E)) < tol:
                break
        abs_E = abs(E)

    return coeffs, abs(E), x


def horner_eval(coeffs, x):
    """Evaluate polynomial sum c_i * x^i using Horner's method."""
    result = np.full_like(x, coeffs[-1], dtype=float)
    for c in coeffs[-2::-1]:
        result = result * x + c
    return result


# =============================================================================
# Part 2: Target Function and Error Analysis
# =============================================================================

def make_target(lambda_val, theta=1.0):
    """Create target function f(x) = theta * exp(-lambda * x)."""
    def f(x):
        return theta * np.exp(-lambda_val * np.asarray(x))
    return f


def compute_approximation_errors(lambda_vals, ds, M=255.0, theta=1.0):
    """
    Compute L∞ and L2 errors for various λ and polynomial degrees d.
    """
    grid = np.linspace(0, M, 10001)

    results = []
    for lam in lambda_vals:
        f = make_target(lam, theta)
        f_true = f(grid)

        for d in ds:
            try:
                coeffs, linf_err, alt_points = remez_approx(f, 0, M, d)
            except np.linalg.LinAlgError:
                results.append((lam, d, np.nan, np.nan, np.nan))
                continue

            p_vals = horner_eval(coeffs, grid)
            abs_err = np.abs(f_true - p_vals)
            l2_err = np.sqrt(np.mean(abs_err ** 2))
            max_err = np.max(abs_err)

            results.append((lam, d, max_err, l2_err, linf_err))

    return results


def plot_error_analysis(lambda_vals, ds, M=255.0):
    """Plot approximation error analysis."""
    results = compute_approximation_errors(lambda_vals, ds, M)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: L∞ error vs d for different λ
    ax = axes[0]
    for lam in lambda_vals:
        linf_errs = []
        ds_used = []
        for r in results:
            if r[0] == lam and not np.isnan(r[2]):
                ds_used.append(r[1])
                linf_errs.append(r[2])
        ax.semilogy(ds_used, linf_errs, 'o-', label=f'λ={lam:.4f}')

    ax.axhline(y=1.5e-5, color='red', linestyle='--', alpha=0.5, label='16-bit quant error')
    ax.axhline(y=1e-7, color='gray', linestyle='--', alpha=0.5, label='1e-7')
    ax.set_xlabel('Polynomial degree d')
    ax.set_ylabel('L∞ error (max |f - p|)')
    ax.set_title('Remez Approximation Error vs Degree')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: f(x) vs p(x) for best λ
    ax = axes[1]
    lam_example = lambda_vals[len(lambda_vals) // 2]
    f = make_target(lam_example)
    x_plot = np.linspace(0, M, 500)
    f_true = f(x_plot)

    for d in [3, 5, 7, 9]:
        try:
            coeffs, _, _ = remez_approx(f, 0, M, d)
            p_vals = horner_eval(coeffs, x_plot)
            err = np.abs(f_true - p_vals)
            ax.semilogy(x_plot, err, label=f'd={d}')
        except np.linalg.LinAlgError:
            pass

    ax.set_xlabel('x')
    ax.set_ylabel('Absolute error |f(x) - p(x)|')
    ax.set_title(f'Pointwise Error (λ={lam_example:.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('polyattn_error_analysis.png', dpi=150)
    plt.close()
    print("[Part 2] Error analysis plot saved to polyattn_error_analysis.png")

    return results


# =============================================================================
# Part 3: Simulated Attention Softmax
# =============================================================================

def true_softmax(z, axis=-1):
    """Numerically stable softmax."""
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def poly_softmax(z, coeffs, axis=-1):
    """
    Polynomial-based softmax approximation.
    Uses exp(z) ≈ p(z) where p is the Remez polynomial.

    For numerical stability: subtract max before polynomial eval,
    same as standard softmax.
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    z_shifted = z - z_max  # all <= 0

    # exp(z_shifted) ≈ p(z_shifted)  -- but p only covers [a, b]
    # For values outside [a, b], clip or handle separately
    # p is trained on [-M, 0], but after shift z_shifted can be more negative

    # Option A: clip to domain
    # z_clipped = np.clip(z_shifted, -6, 0)  # Remez domain

    # Option B: polynomial eval on all values, exp(-large) ≈ 0 anyway
    p_vals = horner_eval(coeffs, -z_shifted)  # exp(z) = p(-z) if p fits exp(-x)
    # Handle very negative z: clip small values
    p_vals = np.maximum(p_vals, 1e-30)

    return p_vals / np.sum(p_vals, axis=axis, keepdims=True)


def test_attention_simulation(seq_len=128, d_head=64, n_heads=1, seed=42):
    """
    Simulate a single attention head computation,
    comparing true softmax vs polynomial softmax output.
    """
    np.random.seed(seed)

    # Random Q, K, V tensors
    Q = np.random.randn(seq_len, d_head).astype(np.float64)
    K = np.random.randn(seq_len, d_head).astype(np.float64)
    V = np.random.randn(seq_len, d_head).astype(np.float64)

    # Attention scores
    scores = Q @ K.T / np.sqrt(d_head)  # [seq, seq]

    # True attention output
    attn_true = true_softmax(scores, axis=-1)
    out_true = attn_true @ V

    # ---- Polynomial attention ----
    # The input to exp is scores / sqrt(d) after subtracting max
    # Domain: shifted z ∈ [-6, 0] typically (N(0,1) after normalization)
    # We approximate exp(x) for x ∈ [-6, 0] via Remez on exp(-y) for y ∈ [0, 6]

    M = 6.0
    coeffs_exp_neg = {}
    for d in [3, 5, 7, 9]:
        try:
            f = make_target(1.0)  # exp(-y) for y in [0, M]
            c, err, _ = remez_approx(f, 0, M, d)
            coeffs_exp_neg[d] = c
        except np.linalg.LinAlgError:
            pass

    results_attn = {}
    for d, c in coeffs_exp_neg.items():
        # poly_exp(x) = horner_eval(c, -x) for x <= 0
        def poly_exp(x):
            return horner_eval(c, -np.asarray(x))

        z_max = np.max(scores, axis=-1, keepdims=True)
        z_shifted = scores - z_max

        # Evaluate polynomial approximation
        p_exp = poly_exp(z_shifted)
        p_exp = np.maximum(p_exp, 1e-30)

        attn_poly = p_exp / np.sum(p_exp, axis=-1, keepdims=True)
        out_poly = attn_poly @ V

        # Error metrics
        attn_diff = np.abs(attn_true - attn_poly)
        out_diff = np.abs(out_true - out_poly)

        results_attn[d] = {
            'attn_max_err': float(np.max(attn_diff)),
            'attn_mean_err': float(np.mean(attn_diff)),
            'output_max_err': float(np.max(out_diff)),
            'output_mean_err': float(np.mean(out_diff)),
            'output_rel_err': float(np.mean(out_diff) / (np.mean(np.abs(out_true)) + 1e-10)),
            'coeffs': c,
        }

    return results_attn, scores, out_true


def plot_attention_results(results_attn):
    """Plot attention simulation results."""
    ds = sorted(results_attn.keys())
    attn_max = [results_attn[d]['attn_max_err'] for d in ds]
    attn_mean = [results_attn[d]['attn_mean_err'] for d in ds]
    out_max = [results_attn[d]['output_max_err'] for d in ds]
    out_mean = [results_attn[d]['output_mean_err'] for d in ds]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(ds))
    w = 0.2
    ax.bar(x - 1.5*w, attn_max, w, label='Attention weights (max)')
    ax.bar(x - 0.5*w, attn_mean, w, label='Attention weights (mean)')
    ax.bar(x + 0.5*w, out_max, w, label='Output (max)')
    ax.bar(x + 1.5*w, out_mean, w, label='Output (mean)')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in ds])
    ax.set_ylabel('Absolute error')
    ax.set_title('PolyAttn: Attention Simulation Error (128 seq, 64 head)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('polyattn_attention_sim.png', dpi=150)
    plt.close()
    print("[Part 3] Attention simulation plot saved to polyattn_attention_sim.png")


# =============================================================================
# Part 4: Error Propagation through Residual Layers
# =============================================================================

def simulate_error_propagation(n_layers=40, seq_len=128, d_model=5120, residual_ratio=0.15, seed=123):
    """
    Simulate error propagation through L residual layers.

    Model: x_{l+1} = x_l + F_l(x_l)
    where F_l is attention + MLP block (the "correction").
    residual_ratio = |F_l| / |x_l| (empirically ~0.1-0.2).

    We inject approximation error ε at each attention layer,
    and track how the total error accumulates.
    """
    np.random.seed(seed)

    # Estimate per-layer attention error from d=7 polynomial
    # From Part 3 results, attention output error ~ 1e-6 to 1e-4
    # We'll use a range of possible values

    eps_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    result = {}

    for eps in eps_values:
        x = np.random.randn(seq_len, d_model).astype(np.float64)  # initial input
        x_norm = np.linalg.norm(x)

        errors_per_layer = []
        for layer in range(n_layers):
            # Correction term (attention + MLP)
            correction = np.random.randn(seq_len, d_model).astype(np.float64)
            correction = correction / np.linalg.norm(correction) * x_norm * residual_ratio

            # Inject error into correction
            correction_noisy = correction + np.random.randn(*correction.shape) * eps * np.linalg.norm(correction)

            # Residual connection
            x_new = x + correction_noisy
            x_norm_new = np.linalg.norm(x_new)

            # Track relative error
            rel_change = np.linalg.norm(correction_noisy - correction) / (x_norm + 1e-10)
            errors_per_layer.append(rel_change * residual_ratio)

            x = x_new
            x_norm = x_norm_new

        # Total relative drift
        total_drift = np.linalg.norm(x - np.random.randn(seq_len, d_model))  # Not meaningful
        # Better: track accumulated relative error
        result[eps] = {
            'per_layer_errors': errors_per_layer,
            'total_relative_error': float(np.sum(errors_per_layer)),
            'rms_accumulated': float(np.sqrt(np.sum([e**2 for e in errors_per_layer]))),
        }

    return result


def plot_error_propagation(prop_results):
    """Plot error propagation results."""
    eps_vals = sorted(prop_results.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for eps in eps_vals:
        errs = prop_results[eps]['per_layer_errors']
        ax.plot(range(1, len(errs)+1), errs, 'o-', markersize=3, label=f'ε={eps:.0e}')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Relative error per layer')
    ax.set_title('Error Propagation through 40 Residual Layers')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('polyattn_error_propagation.png', dpi=150)
    plt.close()
    print("[Part 4] Error propagation plot saved to polyattn_error_propagation.png")


# =============================================================================
# Part 5: Real LLaMA Analysis (if weights available)
# =============================================================================

def analyze_llama_attention_distribution():
    """
    Analyze the actual input distribution to Softmax in LLaMA attention.

    Without real weights, we use the theoretical analysis:
    - Q and K are RMSNorm normalized → each element ~ N(0,1)
    - QK^T / sqrt(d_head): d_head independent N(0,1) products → variance = d_head
    - Dividing by sqrt(d_head): variance = 1
    - After causal mask: valid positions get N(0,1), masked get -inf

    Returns statistics about the expected input range.
    """
    np.random.seed(456)
    d_head = 128
    seq_len = 2048

    # Simulated normalized Q, K
    Q = np.random.randn(seq_len, d_head).astype(np.float64)
    K = np.random.randn(seq_len, d_head).astype(np.float64)

    scores = Q @ K.T / np.sqrt(d_head)

    print("\n[Part 5] Attention score distribution analysis:")
    print(f"  d_head = {d_head}, seq_len = {seq_len}")
    print(f"  Score mean:     {np.mean(scores):.6f}")
    print(f"  Score std:      {np.std(scores):.6f}")
    print(f"  Score min:      {np.min(scores):.4f}")
    print(f"  Score max:      {np.max(scores):.4f}")
    print(f"  |score| < 1:    {np.mean(np.abs(scores) < 1)*100:.2f}%")
    print(f"  |score| < 2:    {np.mean(np.abs(scores) < 2)*100:.2f}%")
    print(f"  |score| < 3:    {np.mean(np.abs(scores) < 3)*100:.2f}%")
    print(f"  |score| < 4:    {np.mean(np.abs(scores) < 4)*100:.2f}%")
    print(f"  |score| < 5:    {np.mean(np.abs(scores) < 5)*100:.2f}%")
    print(f"  |score| < 6:    {np.mean(np.abs(scores) < 6)*100:.2f}%")

    # After subtracting max per row
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    print(f"\n  After subtracting row max:")
    print(f"  Shifted min:    {np.min(scores_shifted):.4f}")
    print(f"  Shifted max:    0.0 (by definition)")
    print(f"  Shifted < -6:   {np.mean(scores_shifted < -6)*100:.4f}%")

    return scores, scores_shifted


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PolyAttn Validation Suite")
    print("=" * 60)

    # ---- Part 2: Error Analysis ----
    print("\n>>> Part 2: Remez Approximation Error Analysis")
    lambda_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    ds = [3, 5, 7, 9, 11]
    error_results = plot_error_analysis(lambda_vals, ds, M=6.0)

    print("\n  λ        d    L∞ error        L2 error")
    print("  " + "-" * 45)
    for lam, d, linf, l2, _ in error_results:
        if not np.isnan(linf):
            print(f"  {lam:<8.4f} {d}    {linf:<14.2e} {l2:<14.2e}")

    # ---- Part 3: Attention Simulation ----
    print("\n>>> Part 3: Simulated Attention Softmax")
    results_attn, _, _ = test_attention_simulation(seq_len=128, d_head=64)
    plot_attention_results(results_attn)

    print("\n  d     Attn MaxErr    Attn MeanErr   Output MaxErr  Output MeanErr  Output RelErr")
    print("  " + "-" * 80)
    for d in sorted(results_attn.keys()):
        r = results_attn[d]
        print(f"  {d}     {r['attn_max_err']:<14.2e} {r['attn_mean_err']:<14.2e} "
              f"{r['output_max_err']:<14.2e} {r['output_mean_err']:<14.2e} "
              f"{r['output_rel_err']:<14.2e}")

    # ---- Part 4: Error Propagation ----
    print("\n>>> Part 4: Error Propagation through 40 Layers")
    prop_results = simulate_error_propagation(n_layers=40, seq_len=128, d_model=5120)
    plot_error_propagation(prop_results)

    print("\n  Per-layer ε    Accumulated (sum)   RMS Accumulated")
    print("  " + "-" * 55)
    for eps in sorted(prop_results.keys()):
        r = prop_results[eps]
        print(f"  {eps:<14.0e} {r['total_relative_error']:<20.8f} {r['rms_accumulated']:<20.8f}")

    # ---- Part 5: LLM Input Distribution ----
    print("\n>>> Part 5: Theoretical Attention Score Distribution")
    analyze_llama_attention_distribution()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary for PolyAttn Paper")
    print("=" * 60)

    # Best result: d=7, typical λ
    d_target = 7
    lam_typical = 1.0  # λ = 1/(γ*sqrt(d)) * B^(k), for mid segments

    f = make_target(lam_typical)
    coeffs_d7, linf_d7, _ = remez_approx(f, 0, 6.0, d_target)

    attn_output_err = results_attn.get(d_target, {}).get('output_rel_err', np.nan)

    print(f"""
    Key numbers (d={d_target}, exp(-x) on [0, 6.0]):
      - Remez L∞ error:    {linf_d7:.2e}
      - 16-bit quant error: 1.5e-5
      - PolyAttn error is {'BELOW' if linf_d7 < 1.5e-5 else 'ABOVE'} quantization error

    Attention simulation (128 seq, 64 head):
      - Output relative error: {attn_output_err:.2e}

    Error propagation (40 layers):
      - With ε={1e-6:.0e} per-layer attention error:
        Total accumulated: {prop_results[1e-6]['total_relative_error']:.6f}
    """)

    print("Plots saved: polyattn_error_analysis.png, polyattn_attention_sim.png, polyattn_error_propagation.png")
    print("Done.")
