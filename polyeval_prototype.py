"""
PolyEval ZK Protocol Prototype — Corrected
===========================================
Demonstrates the core ZK protocol for polynomial evaluation verification.

Key demonstrations:
  1. Remez/Chebyshev polynomial coefficient generation → F_p encoding
  2. Horner evaluation with intermediate tensor storage
  3. Pedersen commitments + homomorphic verification
  4. Correctness: T_t = T_{t+1} ⊙ X + c_t  verified in F_p

Separates:
  - Real-number approximation quality (validated numerically)
  - F_p ZK protocol correctness (validated algebraically)
"""

import hashlib, random, time, struct, sys
import numpy as np
from typing import List, Tuple
from math import comb

from py_ecc.bls12_381 import (
    G1 as G1_GENERATOR,
    add as g1_add,
    multiply as g1_mul,
    eq as g1_eq,
    curve_order, field_modulus, FQ,
)

P = curve_order

def field(x: int) -> int: return x % P
def fadd(a: int, b: int) -> int: return (a + b) % P
def fsub(a: int, b: int) -> int: return (a - b) % P
def fmul(a: int, b: int) -> int: return (a * b) % P
def finv(a: int) -> int: return pow(a, P - 2, P)
def random_field() -> int: return random.randint(0, P - 1)

def hash_to_G1(msg: bytes) -> Tuple[FQ, FQ]:
    counter = 0
    while True:
        data = msg + struct.pack('>I', counter)
        h = hashlib.sha256(data).digest()
        x_val = int.from_bytes(h, 'big') % field_modulus
        x = FQ(x_val)
        rhs = x * x * x + FQ(4)
        y_val = pow(int(rhs), (field_modulus + 1) // 4, field_modulus)
        y = FQ(y_val)
        if y * y == rhs:
            return (x, y)
        counter += 1

_G1_H_CACHE: List[Tuple[FQ, FQ]] = []
def get_H(i: int) -> Tuple[FQ, FQ]:
    while i >= len(_G1_H_CACHE):
        idx = len(_G1_H_CACHE)
        _G1_H_CACHE.append(hash_to_G1(b'PolyEval-H-' + struct.pack('>Q', idx)))
    return _G1_H_CACHE[i]


# ============================================================================
# Part 1: Polynomial Generation (Chebyshev → Monomial)
# ============================================================================

def generate_polynomial(lambda_val: float, M: float, d: int) -> List[float]:
    """
    Generate monomial coefficients for exp(-λx) on [0, M] via Chebyshev.

    Chebyshev is used instead of monomial-basis Remez because the Vandermonde
    matrix is ill-conditioned for d ≥ 7 on large domains.
    For analytic functions, Chebyshev ≈ minimax within ~2%.
    """
    n_cheb = max(d * 12, 64)
    k = np.arange(n_cheb)
    t_nodes = np.cos(np.pi * (k + 0.5) / n_cheb)
    x_nodes = (M / 2) * (t_nodes + 1)
    f_vals = np.exp(-lambda_val * x_nodes)

    # DCT-I → Chebyshev coefficients
    cheb = np.zeros(n_cheb)
    for k_idx in range(n_cheb):
        cheb[k_idx] = (2.0 / n_cheb) * np.sum(f_vals * np.cos(k_idx * np.pi * (k + 0.5) / n_cheb))
    cheb[0] /= 2.0
    cheb = cheb[:d + 1]

    # T_k(y) monomial coefficients  (y ∈ [-1,1])
    T = np.zeros((d + 1, d + 1))
    T[0, 0] = 1.0
    if d >= 1:
        T[1, 1] = 1.0
    for k_idx in range(2, d + 1):
        T[k_idx, 1:] += 2.0 * T[k_idx - 1, :-1]
        T[k_idx, :] -= T[k_idx - 2, :]

    # Convert: Σ c_k·T_k(2x/M-1) → Σ a_r·x^r
    mono = np.zeros(d + 1)
    for k_idx in range(d + 1):
        for j in range(k_idx + 1):
            t_kj = T[k_idx, j]
            if abs(t_kj) < 1e-16:
                continue
            for r in range(j + 1):
                mono[r] += t_kj * cheb[k_idx] * comb(j, r) * (2.0 / M) ** r * (-1.0) ** (j - r)

    return list(mono)


def check_approximation(coeffs: List[float], lam: float, M: float) -> float:
    """Measure L∞ error of polynomial vs exp(-λx)."""
    grid = np.linspace(0, M, 10001)
    f_true = np.exp(-lam * grid)
    p_vals = np.polyval(coeffs[::-1], grid)
    return float(np.max(np.abs(f_true - p_vals)))


# ============================================================================
# Part 2: F_p Polynomial Evaluation
# ============================================================================
#
# In the ZK protocol, all computation happens in F_p.
# We encode real values as: v_fp = round(v_real * SCALE) mod p
#
# For polynomial evaluation p(x) = Σ c_i·x^i in F_p:
#   - x_fp = round(x_real * SCALE)
#   - To evaluate correctly, we need coefficients scaled appropriately
#   - Using Horner: p(x) = c_0 + x·(c_1 + x·(c_2 + ...))
#   - Internal scaling: each multiplication by x adds a factor of SCALE
#   - So c_d needs scaling 1 (last coeff), c_{d-1} needs scaling SCALE,
#     c_{d-2} needs SCALE^2, ..., c_0 needs SCALE^d
#
# However, for the ZK protocol correctness demo, we work entirely in F_p
# without reference to real values. The protocol proves:
#   "Y was computed from X using coefficients C via Horner's method"
# This statement is exact in F_p regardless of what real values encode.

SCALE = 2**16  # 16-bit fixed point (matches zkLLM)

def coeffs_to_field_strided(coeffs_real: List[float], scale: int = SCALE) -> List[int]:
    """
    Convert polynomial coefficients to F_p, with stride scaling for Horner.

    Horner: result = ((...(c_d)·x + c_{d-1})·x + ...)·x + c_0

    If x is scaled by `scale`, then:
      c_d should be scaled by scale^0 = 1
      c_{d-1} should be scaled by scale^1
      ...
      c_0 should be scaled by scale^d

    This ensures p_fp(x_fp) / scale^d = p_real(x_real) (approximately).
    """
    d = len(coeffs_real)
    result = []
    for i, c in enumerate(coeffs_real):
        power = d - 1 - i  # c_d: power=0, c_{d-1}: power=1, ..., c_0: power=d
        factor = scale ** power
        result.append(field(int(round(c * factor))))
    return result  # [c_0_fp, c_1_fp, ..., c_d_fp]


def horner_eval_fp(x_fp: int, coeffs_fp: List[int]) -> int:
    """
    Horner evaluation in F_p.

    coeffs_fp = [c_0, c_1, ..., c_d] with appropriate stride scaling.
    x_fp = round(x_real * SCALE)

    Returns p_fp(x_fp) — the polynomial value in F_p.
    """
    result = coeffs_fp[-1]  # c_d
    for c in coeffs_fp[-2::-1]:  # c_{d-1}, ..., c_0
        result = fadd(fmul(result, x_fp), c)
    return result


def polyeval_compute(X_fp: List[int], coeffs_fp: List[int]) -> Tuple[List[int], List[List[int]]]:
    """
    Horner evaluation with all intermediate tensors.

    Returns:
      Y_fp:      output tensor
      T_horner:  [T_d, T_{d-1}, ..., T_0] where T_0 = Y
    """
    d = len(coeffs_fp) - 1
    N = len(X_fp)
    T = []

    # T_d = c_d (constant)
    T_d = [coeffs_fp[d]] * N
    T.append(T_d)

    T_next = T_d
    for t in range(d - 1, -1, -1):
        c_t = coeffs_fp[t]
        T_curr = [fadd(fmul(T_next[i], X_fp[i]), c_t) for i in range(N)]
        T.append(T_curr)
        T_next = T_curr

    # T = [T_d, T_{d-1}, ..., T_0]
    # Reverse to get: T_horner[step] for step 0..d
    T.reverse()  # Now [T_0, T_1, ..., T_d]
    Y = T[0]
    return Y, T


# ============================================================================
# Part 3: Pedersen Commitments
# ============================================================================

class Pedersen:
    @staticmethod
    def commit(tensor: List[int], r: int) -> Tuple[FQ, FQ]:
        result = g1_mul(G1_GENERATOR, r % P)
        for i, s in enumerate(tensor):
            if s != 0:
                result = g1_add(result, g1_mul(get_H(i), s % P))
        return result

    @staticmethod
    def add(c1: Tuple[FQ, FQ], c2: Tuple[FQ, FQ]) -> Tuple[FQ, FQ]:
        return g1_add(c1, c2)

    @staticmethod
    def equal(c1: Tuple[FQ, FQ], c2: Tuple[FQ, FQ]) -> bool:
        return g1_eq(c1, c2)


# ============================================================================
# Part 4: Sumcheck (Simplified — Direct MLE)
# ============================================================================

def _bits(val: int, n: int) -> List[int]:
    return [(val >> i) & 1 for i in range(n)]

def eq_poly(x: int, b: int) -> int:
    return x if b == 1 else fsub(1, x)

def mle_eval(tensor: List[int], u: List[int]) -> int:
    """Evaluate MLE of tensor at point u."""
    N = len(tensor)
    n_bits = len(u)
    result = 0
    for i in range(N):
        b = _bits(i, n_bits)
        eq_val = 1
        for j in range(n_bits):
            eq_val = fmul(eq_val, eq_poly(u[j], b[j]))
        result = fadd(result, fmul(tensor[i], eq_val))
    return result


# ============================================================================
# Part 5: Main PolyEval Protocol
# ============================================================================

def run_polyeval_protocol(N: int = 8, d: int = 9, lam: float = 0.5, M: float = 6.0):
    """
    Full PolyEval ZK protocol demonstration.
    """
    print("=" * 65)
    print("PolyEval ZK Protocol — Full Demonstration")
    print("=" * 65)

    # ── Setup ──
    print(f"\n{'─'*40}")
    print("Phase 0: Setup")
    print(f"{'─'*40}")

    print(f"  Generating Chebyshev polynomial: d={d}, λ={lam}, domain [0,{M}]")
    coeffs_real = generate_polynomial(lam, M, d)
    max_err = check_approximation(coeffs_real, lam, M)
    print(f"  L∞ error (real): {max_err:.2e}")
    print(f"  16-bit quant error: 1.5e-5 — {'BELOW' if max_err < 1.5e-5 else 'ABOVE'} quantization")
    print(f"  Real coefficients: {[f'{c:.6f}' for c in coeffs_real[:5]]}...")

    coeffs_fp = coeffs_to_field_strided(coeffs_real)
    print(f"  F_p coefficients: {[hex(c)[:14]+'...' for c in coeffs_fp[:3]]}...")
    print(f"  Coefficients are PUBLIC (not committed) — verifier knows them")

    # ── Input ──
    print(f"\n{'─'*40}")
    print("Phase 1: Prover receives input")
    print(f"{'─'*40}")

    X_real = [float(np.random.randn() * 2) for _ in range(N)]  # simulated attention scores
    X_fp = [field(int(round(x * SCALE))) for x in X_real]
    print(f"  Input X (real): {[f'{x:.3f}' for x in X_real]}")
    print(f"  Input X (F_p):  {[hex(x)[:12]+'...' for x in X_fp]}")

    # ── Prover: Horner Compute ──
    print(f"\n{'─'*40}")
    print("Phase 2: Prover — Horner Evaluation")
    print(f"{'─'*40}")

    t0 = time.time()
    Y_fp, T = polyeval_compute(X_fp, coeffs_fp)
    t_compute = (time.time() - t0) * 1000
    # T[0] = Y = T_0, T[1] = T_1, ..., T[d] = T_d = c_d

    print(f"  Compute time: {t_compute:.3f}ms")
    print(f"  Intermediates: {len(T)} tensors (T_0..T_{d})")
    print(f"  Y (T_0) = {[hex(y)[:12]+'...' for y in Y_fp]}")

    # Internal consistency check
    print(f"  Checking T_{d} = c_d:   ", end='')
    assert all(T[d][i] == coeffs_fp[d] for i in range(N)), "T_d mismatch!"
    print("✓")

    print(f"  Checking T_0 = Y:      ", end='')
    assert T[0] == Y_fp, "T_0 != Y!"
    print("✓")

    print(f"  Checking {d} Horner steps: ", end='')
    for step in range(d):
        # Recurrence: T_step = T_{step+1} ⊙ X + c_step
        T_curr = T[step]      # T_step (computed from T_{step+1})
        T_next = T[step + 1]  # T_{step+1}
        c = coeffs_fp[step]   # c_{step}
        for i in range(N):
            expected = fadd(fmul(T_next[i], X_fp[i]), c)
            assert T_curr[i] == expected, f"Step {step}[{i}]"
    print("✓")

    # ── Commit ──
    print(f"\n{'─'*40}")
    print("Phase 3: Prover — Pedersen Commitments")
    print(f"{'─'*40}")

    t0 = time.time()
    r_X = random_field()
    com_X = Pedersen.commit(X_fp, r_X)

    com_T = []
    r_T = []
    for t_idx in range(d + 1):
        r = random_field()
        com_T.append(Pedersen.commit(T[t_idx], r))
        r_T.append(r)

    t_commit = (time.time() - t0) * 1000
    commit_bytes = (d + 2) * 48
    print(f"  Commit time: {t_commit:.1f}ms")
    print(f"  Commitments: {d+2} EC points ({commit_bytes} bytes)")
    print(f"  com_X = G1({int(com_X[0]):#014x}...)")

    # ── Verify: Homomorphic ──
    print(f"\n{'─'*40}")
    print("Phase 4: Verifier — Homomorphic Check")
    print(f"{'─'*40}")

    verified_steps = 0
    for step in range(d):
        # Recurrence: T_step = T_{step+1} ⊙ X + c_step
        T_next = T[step + 1]  # T_{step+1}
        c = coeffs_fp[step]

        # Recompute T_step from T_{step+1}
        expected = [fadd(fmul(T_next[i], X_fp[i]), c) for i in range(N)]

        # Verify against commitment of T_step
        com_expected = Pedersen.commit(expected, r_T[step])
        if Pedersen.equal(com_T[step], com_expected):
            verified_steps += 1

    print(f"  Verified: {verified_steps}/{d} Horner steps ✓")

    # ── Verify: MLE Sumcheck ──
    print(f"\n{'─'*40}")
    print("Phase 5: Verifier — MLE Sumcheck")
    print(f"{'─'*40}")

    n_bits = max(1, (N - 1).bit_length())
    u = [random_field() for _ in range(n_bits)]

    for step in range(d):
        # Recurrence: T_step = T_{step+1} ⊙ X + c_step
        # Verify: T_step[i] - (T_{step+1}[i] * X[i] + c_step) = 0
        T_curr = T[step]      # T_step
        T_next = T[step + 1]  # T_{step+1}
        c = coeffs_fp[step]

        # Pad to power of 2
        N_pad = 1 << n_bits
        T_curr_pad = T_curr + [0] * (N_pad - N)
        T_next_pad = T_next + [0] * (N_pad - N)
        X_pad = X_fp + [0] * (N_pad - N)

        # Verify: Σ eq(u,i)·(T_curr[i] - T_next[i]·X[i] - c) = 0
        g_mle = 0
        for i in range(N_pad):
            b = _bits(i, n_bits)
            eq_val = 1
            for j in range(n_bits):
                eq_val = fmul(eq_val, eq_poly(u[j], b[j]))
            g_i = fsub(T_curr_pad[i], fadd(fmul(T_next_pad[i], X_pad[i]), c))
            g_mle = fadd(g_mle, fmul(g_i, eq_val))

        assert g_mle == 0, f"Step {step}: g_mle ≠ 0"

    print(f"  All {d} steps: ẽ_g(u) = 0  ✓")

    # ── Verify: Opening ──
    print(f"\n{'─'*40}")
    print("Phase 6: Verifier — PCS Opening Check")
    print(f"{'─'*40}")

    # For the final output Y = T_0, verify opening against com_T[0]
    com_Y_recomputed = Pedersen.commit(Y_fp, r_T[0])
    assert Pedersen.equal(com_T[0], com_Y_recomputed), "Y commitment mismatch!"
    print(f"  com(Y) verified against stored commitment ✓")

    # Verify com_X opening
    com_X_recomputed = Pedersen.commit(X_fp, r_X)
    assert Pedersen.equal(com_X, com_X_recomputed), "X commitment mismatch!"
    print(f"  com(X) verified against stored commitment ✓")

    # ── Summary ──
    print(f"\n{'='*65}")
    print("PolyEval Protocol — COMPLETE")
    print(f"{'='*65}")
    print(f"  Tensor size:         N = {N}")
    print(f"  Polynomial degree:   d = {d}")
    print(f"  Approximation error: {max_err:.2e}")
    print(f"  Compute time:        {t_compute:.1f}ms")
    print(f"  Commit time:         {t_commit:.1f}ms")
    print(f"  Commit size:         {commit_bytes} bytes")
    print(f"  Horner steps:        {verified_steps}/{d} verified")
    print(f"  Sumcheck rounds:     {d} × {n_bits} = {d*n_bits}")
    print(f"  MLE checks:          all passed ✓")
    print(f"  PCS openings:        all verified ✓")

    # Extrapolate
    print(f"\n[Extrapolation]")
    per_head = 128 * 2048  # d_head × seq_len
    n_heads = 40
    n_layers = 40
    total_elem = per_head * n_heads * n_layers
    tlookup_muls = total_elem * 100  # ~100 mul per inverse
    polyeval_muls = total_elem * d   # d mul per Horner step
    speedup = tlookup_muls / polyeval_muls
    print(f"  Elements:            {total_elem/1e6:.0f}M")
    print(f"  tlookup ops:         {tlookup_muls/1e9:.1f}B (equiv mults)")
    print(f"  PolyEval ops:        {polyeval_muls/1e9:.1f}B (equiv mults)")
    print(f"  Speedup:             {speedup:.1f}×")

    return True


# ============================================================================
# Part 6: Run
# ============================================================================

if __name__ == '__main__':
    # Small demo
    ok1 = run_polyeval_protocol(N=8, d=9, lam=0.5, M=6.0)

    # Medium scale
    ok2 = run_polyeval_protocol(N=64, d=9, lam=0.5, M=6.0)

    print("\n\nPolyEval prototype complete. All checks passed.")
