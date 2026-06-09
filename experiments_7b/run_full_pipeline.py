"""
Full ZK Proof Pipeline — Chinese-Llama-2-7b
============================================
Adapts zkLLM scripts for our local model.
Step 1: ppgen → Step 2: commit-param → Step 3: self-attn (linear + attn)
"""
import os, sys, subprocess, time, json
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = '/root/autodl-tmp/chinese-llama-2-7b'
WORKDIR = '/root/autodl-tmp/zkllm-workdir/Llama-2-7b'
ZKDIR = '/root/autodl-tmp/zkllm-ccs2024'

def run_cmd(cmd, desc='', timeout=120):
    print(f'  [{desc}] {cmd[:120]}...' if len(cmd) > 120 else f'  [{desc}] {cmd}')
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    ok = r.returncode == 0
    if ok:
        print(f'    OK ({dt:.1f}s)')
    else:
        print(f'    FAILED ({dt:.1f}s)')
        if r.stderr: print(f'    STDERR: {r.stderr[-300:]}')
    return ok, dt, r.stdout, r.stderr

# ================================================================
# Step 1: Public Parameters (ppgen)
# ================================================================
def step_ppgen(model):
    print(f'\n{\"=\"*50}')
    print('STEP 1: Public Parameter Generation (ppgen)')
    print(f'{\"=\"*50}')

    os.chdir(ZKDIR)
    log_off = 5
    layer0 = model.model.layers[0]
    pp_files = []

    # ppgen generates one PP file per parameter SHAPE (all layers share same shape)
    for name, w in layer0.named_parameters():
        if len(w.shape) == 2:
            pp_size = w.shape[0] << log_off
        elif len(w.shape) == 1:
            pp_size = w.shape[0]
        else:
            continue
        pp_path = f'{WORKDIR}/{name}-pp.bin'
        ok, dt, _, _ = run_cmd(f'./ppgen {pp_size} {pp_path}', f'ppgen {name}', timeout=30)
        if ok: pp_files.append(name)

    print(f'  Generated {len(pp_files)} PP files')
    return pp_files

# ================================================================
# Step 2: Commit Model Weights
# ================================================================
def step_commit(model):
    print(f'\n{\"=\"*50}')
    print('STEP 2: Model Weight Commitment (commit-param)')
    print('         This commits ALL 32 layers x 7 params = 224 params')
    print(f'{\"=\"*50}')

    os.chdir(ZKDIR)
    scaling_factor = 1 << 16
    n_layers = len(model.model.layers)
    total_params = 0
    success_params = 0
    commit_start = time.time()

    for i, layer in enumerate(model.model.layers):
        layer_start = time.time()
        for name, w in layer.named_parameters():
            total_params += 1
            if len(w.shape) == 2:
                w_orig = w.float().T  # transpose for zkLLM format
            else:
                w_orig = w.float()

            w_int = torch.round(w_orig * scaling_factor).to(torch.int32)
            max_diff = ((w_int.float() / scaling_factor) - w_orig).abs().max().item()

            pp_path = f'{WORKDIR}/{name}-pp.bin'
            int_path = f'{WORKDIR}/layer-{i}-{name}-int.bin'
            com_path = f'{WORKDIR}/layer-{i}-{name}-commitment.bin'

            # Save int weights
            w_int.cpu().detach().numpy().astype(np.int32).tofile(int_path)

            if len(w_int.shape) == 2:
                shape_str = f'{w_int.shape[0]} {w_int.shape[1]}'
            else:
                shape_str = f'{w_int.shape[0]} 1'

            ok, dt, _, _ = run_cmd(
                f'./commit-param {pp_path} {int_path} {com_path} {shape_str}',
                f'L{i}/{name} (diff={max_diff:.2e})', timeout=120)
            if ok: success_params += 1

        layer_dt = time.time() - layer_start
        # Estimate
        if i == 1:
            est = layer_dt * n_layers
            print(f'  Estimated remaining: {est * (n_layers - i) / n_layers / 60:.0f} min')

        if i % 8 == 0:
            print(f'  Layer {i}/{n_layers} done ({layer_dt:.0f}s)')

    total_dt = time.time() - commit_start
    print(f'\n  Commit complete: {success_params}/{total_params} OK ({total_dt:.0f}s = {total_dt/60:.1f}min)')
    return success_params == total_params

# ================================================================
# Step 3: Self-Attention Proof
# ================================================================
def step_self_attn(model):
    print(f'\n{\"=\"*50}')
    print('STEP 3: Self-Attention ZK Proof (linear + attn)')
    print(f'{\"=\"*50}')

    os.chdir(ZKDIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    layer_idx = 0
    seq_len = 4  # tiny test first
    embed_dim = model.model.layers[0].self_attn.q_proj.in_features
    layer_prefix = f'layer-{layer_idx}'
    scaling_factor = 1 << 16

    # Create input
    input_ids = torch.randint(0, min(tokenizer.vocab_size, 32000), (1, seq_len))
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        hidden_int = (hidden[0] * scaling_factor).round().clamp(-2**31, 2**31-1).to(torch.int32)

    input_path = f'{WORKDIR}/input.bin'
    hidden_int.cpu().numpy().astype(np.int32).tofile(input_path)
    output_path = f'{WORKDIR}/output.bin'
    print(f'  Input: {seq_len} tokens, {embed_dim} dim → {input_path}')

    # Check required files exist
    for prefix in ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight']:
        for suffix in ['-pp.bin']:
            pp = f'{WORKDIR}/{prefix}{suffix}'
            if not os.path.exists(pp):
                print(f'  MISSING: {pp}')
        for suffix in ['-int.bin']:
            f = f'{WORKDIR}/{layer_prefix}-{prefix}{suffix}'
            if not os.path.exists(f):
                # Try without layer prefix
                alt = f'{WORKDIR}/{prefix}{suffix}'
                if os.path.exists(alt):
                    # Create symlink
                    os.symlink(alt, f)
                    print(f'  LINKED: {alt} → {f}')
                else:
                    print(f'  MISSING: {f}')

    # ---- Linear phase ----
    print(f'\n  --- Linear Proof ---')
    linear_ok, linear_dt, out, err = run_cmd(
        f'./self-attn linear {input_path} {seq_len} {embed_dim} {WORKDIR} {layer_prefix} {output_path}',
        'linear', timeout=120)
    print(f'    stdout: {out.strip()[-200:]}')

    if not linear_ok:
        print('  Linear failed. Checking temp files...')
        for f in ['temp_Q.bin', 'temp_K.bin', 'temp_V.bin']:
            print(f'    {f}: {\"EXISTS\" if os.path.exists(f) else \"MISSING\"} (size {os.path.getsize(f) if os.path.exists(f) else 0})')
        return False, 0, 0

    # ---- Attention phase ----
    print(f'\n  --- Attention Proof (Softmax ZK) ---')
    attn_ok, attn_dt, out_a, err_a = run_cmd(
        f'./self-attn attn {input_path} {seq_len} {embed_dim} {WORKDIR} {layer_prefix} {output_path}',
        'attn', timeout=120)
    print(f'    stdout: {out_a.strip()[-300:]}')

    return linear_ok and attn_ok, linear_dt, attn_dt

# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('PolyAttn Full ZK Pipeline — Chinese-Llama-2-7b')
    print('=' * 70)

    os.makedirs(WORKDIR, exist_ok=True)

    print('\nLoading model...')
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True)
    n_layers = len(model.model.layers)
    print(f'  Layers: {n_layers}, d_model: {model.config.hidden_size}')

    # Ask user which steps to run
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=str, default='all', help='Steps: ppgen, commit, attn, all')
    ap.add_argument('--layer', type=int, default=0, help='Layer to test')
    ap.add_argument('--seqlen', type=int, default=4, help='Sequence length')
    args = ap.parse_args()

    results = {}

    if args.step in ['ppgen', 'all']:
        pp_files = step_ppgen(model)
        results['ppgen_files'] = len(pp_files)

    if args.step in ['commit', 'all']:
        ok = step_commit(model)
        results['commit_ok'] = ok

    if args.step in ['attn', 'all']:
        ok, linear_t, attn_t = step_self_attn(model)
        results['linear_ok'] = ok
        results['linear_time'] = linear_t
        results['attn_time'] = attn_t

    print(f'\n{\"=\"*70}')
    print('PIPELINE COMPLETE')
    print(json.dumps(results, indent=2, default=str))
    print(f'{\"=\"*70}')
