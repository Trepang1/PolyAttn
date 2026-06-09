"""Diagnostic: check if F.softmax monkey-patch works on Chinese-Llama-2-7b"""
import torch
import torch.nn.functional as F
import numpy as np

# Test 1: Basic monkey-patch
test_x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
orig = F.softmax(test_x, dim=-1)

def fake_softmax(x, dim=-1, **kw):
    return x / x.sum(dim=dim, keepdim=True)

F.softmax = fake_softmax
patched = F.softmax(test_x, dim=-1)
print(f'Basic monkey-patch works: {not torch.allclose(orig, patched)}')

# Test 2: Check model
from transformers import AutoModelForCausalLM
import json

with open('/root/autodl-tmp/chinese-llama-2-7b/config.json') as f:
    config = json.load(f)
print(f'Model type: {config.get("model_type", "?")}')

model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/chinese-llama-2-7b',
    torch_dtype=torch.bfloat16, device_map='auto',
    trust_remote_code=True, attn_implementation='eager')
model.eval()

print(f'Config attn_implementation: {getattr(model.config, "_attn_implementation", "N/A")}')

# Check attention class
layer0_attn = model.model.layers[0].self_attn
print(f'Attention class: {type(layer0_attn).__name__}')

# Test 3: Does F.softmax get called?
import torch.nn.functional as F_ref
call_count = [0]
orig_f = F_ref.softmax
def counting_softmax(*args, **kwargs):
    call_count[0] += 1
    return orig_f(*args, **kwargs)
F_ref.softmax = counting_softmax
try:
    dummy = torch.randint(0, 1000, (1, 16)).cuda()
    _ = model(dummy)
finally:
    F_ref.softmax = orig_f
print(f'F.softmax called {call_count[0]} times during forward')

if call_count[0] == 0:
    print("DIAGNOSIS: F.softmax is NEVER called! Monkey-patch CANNOT work!")
    print("The model likely uses a fused kernel (SDPA/flash-attn) that bypasses F.softmax")
else:
    print(f"Monkey-patch should work ({call_count[0]} calls per forward)")

# Test 4: Try with a known poly_softmax
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

def horner_torch(c, x):
    y = torch.full_like(x, c[-1])
    for ci in reversed(c[:-1]): y = y * x + ci
    return y

domain_M = 8.0; d = 9
lam_val = 1.0
coeffs = generate_polynomial(lam_val, domain_M, d)
cn = np.array(coeffs, dtype=np.float64)

def poly_fn(logits, dim=-1, dtype=None, **kw):
    x = logits.float(); mx = x.max(dim=dim, keepdim=True).values
    xs = x - mx; xc = torch.clamp(xs, -domain_M, 0.0)
    y = horner_torch(cn, -xc); y = torch.clamp(y, min=1e-30)
    out = y / y.sum(dim=dim, keepdim=True)
    t = dtype if dtype is not None else logits.dtype
    return out.to(t) if t != torch.float32 else out

# Test with a real prompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/chinese-llama-2-7b', trust_remote_code=True)

prompt = "The future of artificial intelligence lies in the development of"
enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
input_ids = enc['input_ids'].cuda()

# Original output
torch.manual_seed(42)
out_orig = model(input_ids)
logits_orig = out_orig.logits.float()

# Poly output
F_ref.softmax = poly_fn
torch.manual_seed(42)
out_poly = model(input_ids)
logits_poly = out_poly.logits.float()
F_ref.softmax = orig_f

diff = (logits_orig - logits_poly).abs()
cos_sim = float(F.cosine_similarity(logits_orig.view(-1), logits_poly.view(-1), dim=0))
print(f'\nReal test: cos_sim={cos_sim:.6f}, max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}')
print(f'Logits IDENTICAL: {torch.allclose(logits_orig, logits_poly)}')

if torch.allclose(logits_orig, logits_poly):
    print("\n*** CONFIRMED: Monkey-patch NOT working! ***")
    print("The model outputs are IDENTICAL with and without poly_softmax.")
else:
    print("\n*** Monkey-patch IS working! Outputs differ. ***")
