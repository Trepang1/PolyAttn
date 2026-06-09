"""Full zkLLM pipeline on A100 — ppgen + commit + self-attn"""
import os, time, torch
from transformers import AutoModelForCausalLM

MODEL = '/root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-7b'
WORK = '/root/autodl-tmp/zkllm-workdir/Llama-2-7b'
ZKDIR = '/root/autodl-tmp/zkllm-ccs2024'
sf = 1 << 16

os.makedirs(WORK, exist_ok=True)
os.chdir(ZKDIR)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True)
L0 = model.model.layers[0]
n_layers = len(model.model.layers)
print(f"Loaded: {n_layers} layers, embed_dim={L0.self_attn.q_proj.in_features}")

# ==== ppgen ====
print("\n=== Step 1: ppgen ===")
for name, w in L0.named_parameters():
    if len(w.shape) == 2: sz = w.shape[0] << 5
    elif len(w.shape) == 1: sz = w.shape[0]
    else: continue
    r = os.system(f"./ppgen {sz} {WORK}/{name}-pp.bin")
    print(f"  {name}: {'OK' if r==0 else 'FAIL'}")

# ==== commit layer 0 ====
print("\n=== Step 2: commit-param (Layer 0) ===")
t0 = time.time()
for name, w in L0.named_parameters():
    w_orig = w.float().T if len(w.shape) == 2 else w.float()
    w_int = torch.round(w_orig * sf).to(torch.int32)
    pp_p = f"{WORK}/{name}-pp.bin"
    int_p = f"{WORK}/layer-0-{name}-int.bin"
    com_p = f"{WORK}/layer-0-{name}-commitment.bin"
    w_int.cpu().detach().numpy().astype("int32").tofile(int_p)
    shape = f"{w_int.shape[0]} {w_int.shape[1]}" if len(w_int.shape) == 2 else f"{w_int.shape[0]} 1"
    r = os.system(f"./commit-param {pp_p} {int_p} {com_p} {shape}")
    print(f"  {name}: {'OK' if r==0 else 'FAIL'}")
print(f"  Layer 0: {time.time()-t0:.0f}s")

# ==== self-attn ====
print("\n=== Step 3: self-attn ===")
seq, emb = 4, L0.self_attn.q_proj.in_features
inp = (torch.randn(seq, emb) * sf).round().clamp(-2**31, 2**31-1).to(torch.int32)
inp_path = f"{WORK}/input.bin"
inp.cpu().numpy().astype("int32").tofile(inp_path)

print("  Linear phase...")
r1 = os.system(f"./self-attn linear {inp_path} {seq} {emb} {WORK} layer-0 {WORK}/out.bin")
print(f"  Linear: {'OK' if r1==0 else 'FAIL (exit '+str(r1)+')'}")

if r1 == 0:
    print("  Attention phase...")
    r2 = os.system(f"./self-attn attn {inp_path} {seq} {emb} {WORK} layer-0 {WORK}/out.bin")
    print(f"  Attention: {'OK' if r2==0 else 'FAIL (exit '+str(r2)+')'}")
    if r2 == 0:
        print("\n*** FULL PIPELINE VERIFIED ***")
    else:
        print("\n*** Attention phase FAILED ***")
else:
    print("\n*** Linear phase FAILED ***")
