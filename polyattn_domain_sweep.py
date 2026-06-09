"""Domain sweep for PolyAttn PPL test."""
import torch, torch.nn.functional as F, numpy as np, time, sys
from math import comb

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

def check_approximation(coeffs, lam, M):
    grid = np.linspace(0, M, 5001)
    f_true = np.exp(-lam * grid)
    p_vals = np.polyval(coeffs[::-1], grid)
    return float(np.max(np.abs(f_true - p_vals)))

def horner_torch(coeffs, x):
    y = torch.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        y = y * x + c
    return y

def make_poly_softmax(domain_M, d=9):
    lam = 1.0
    coeffs = generate_polynomial(lam, domain_M, d)
    linf = check_approximation(coeffs, lam, domain_M)
    coeffs_np = np.array(coeffs, dtype=np.float64)

    def poly_softmax(logits, dim=-1, dtype=None, **kwargs):
        x = logits.float()
        x_max = x.max(dim=dim, keepdim=True).values
        x_shifted = x - x_max
        x_clipped = torch.clamp(x_shifted, -domain_M, 0.0)
        neg_x = -x_clipped
        y = horner_torch(coeffs_np, neg_x)
        y = torch.clamp(y, min=1e-30)
        out = y / y.sum(dim=dim, keepdim=True)
        target = dtype if dtype is not None else logits.dtype
        return out.to(target) if target != torch.float32 else out

    return poly_softmax, linf


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, max_length=256, device='cuda'):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = enc['input_ids'].to(device)
        if input_ids.size(1) < 4: continue
        outputs = model(input_ids, labels=input_ids)
        if outputs.loss is not None and not torch.isnan(outputs.loss):
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    if total_tokens == 0: return float('inf'), float('inf')
    avg = total_loss / total_tokens
    return float(np.exp(avg)), float(avg)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = '/root/autodl-tmp/Qwen2.5-0.5B'
    device = 'cuda'
    texts = [
        "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",
        "深度学习的发展推动了自然语言处理领域的巨大进步。",
        "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",
        "计算机视觉技术使得机器能够理解和分析图像内容。",
        "量子计算作为一种新型计算范式，有望在密码学等领域带来革命性变化。",
        "The future of artificial intelligence lies in developing more efficient algorithms.",
        "Machine learning has revolutionized how we approach complex problems in science.",
        "Natural language processing enables computers to understand human language at scale.",
    ] * 5

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    model.eval()

    # Original PPL
    orig_softmax = F.softmax
    ppl_orig, loss_orig = compute_ppl(model, tokenizer, texts, 256, device)
    print(f"Original PPL: {ppl_orig:.4f} (loss={loss_orig:.4f})")
    print()

    # Sweep domains
    print(f"{'Domain':<12} {'Linf':<12} {'PPL':<12} {'Delta':<12} {'Change%':<10}")
    print("-" * 58)

    domains = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
    results = []

    for dm in domains:
        poly_fn, linf = make_poly_softmax(dm, d=9)
        F.softmax = poly_fn
        t0 = time.time()
        ppl, loss = compute_ppl(model, tokenizer, texts, 256, device)
        dt = time.time() - t0
        delta = ppl - ppl_orig
        print(f"[-{dm:<4.0f},0]     {linf:<12.2e} {ppl:<12.4f} {delta:<+12.4f} {delta/ppl_orig*100:<+10.2f}%")
        results.append((dm, linf, ppl, delta, dt))

    F.softmax = orig_softmax

    # Best domain
    best = min(results, key=lambda x: abs(x[3]))
    print(f"\nBest domain: [-{best[0]}, 0], PPL delta = {best[3]:+.4f}")

    # Test with d=7 too
    print(f"\n--- d=7 comparison ---")
    for dm in [6.0, 8.0, 10.0]:
        poly_fn, linf = make_poly_softmax(dm, d=7)
        F.softmax = poly_fn
        ppl, loss = compute_ppl(model, tokenizer, texts, 256, device)
        delta = ppl - ppl_orig
        print(f"  d=7, [-{dm},0]: L={linf:.2e}, PPL={ppl:.4f}, delta={delta:+.4f}")

    F.softmax = orig_softmax
    print("Done.")


if __name__ == '__main__':
    main()
