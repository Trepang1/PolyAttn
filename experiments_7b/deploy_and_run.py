"""
Deploy experiment scripts to AutoDL server and run them.
Uses paramiko for SSH/SCP.
"""
import paramiko
import os
import sys
import time
import json

# ============================================================================
# Config
# ============================================================================

SSH_HOST = 'connect.bjb2.seetacloud.com'
SSH_PORT = 46569
SSH_USER = 'root'
SSH_PASSWORD = 'dMJn8cOS7K6j'

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = '/root/autodl-tmp/experiments_7b'
RESULT_DIR = '/root/autodl-tmp/experiments_7b_results'

EXPERIMENTS = [
    'exp1_degree_ablation.py',
    'exp2_domain_degree_sweep.py',
    'exp3_long_sequence.py',
    'exp4_wikitext_benchmark.py',
    'exp5_attention_distribution.py',
    'exp6_error_propagation.py',
]

# ============================================================================
# SSH Connection
# ============================================================================

def create_ssh_client():
    """Create SSH client with password auth."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD,
                   look_for_keys=False, allow_agent=False)
    return client

def run_command(ssh, cmd, timeout=300):
    """Run a command and return stdout, stderr, exit_code."""
    print(f"  [CMD] {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    exit_code = stdout.channel.recv_exit_status()
    return out, err, exit_code

# ============================================================================
# Main
# ============================================================================

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print("=" * 70)
    print("PolyAttn 7B Experiments: Deploy & Run")
    print("=" * 70)

    # Connect
    print("\nConnecting to AutoDL server...")
    ssh = create_ssh_client()
    sftp = ssh.open_sftp()
    print("Connected!")

    # Check GPU
    out, err, code = run_command(ssh, "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
    print(f"GPU: {out.strip()}")

    # Check disk
    out, err, code = run_command(ssh, "df -h /root/autodl-tmp | tail -1")
    print(f"Disk: {out.strip()}")

    # Create directories
    run_command(ssh, f"mkdir -p {REMOTE_DIR} {RESULT_DIR}")
    print(f"Created directories on remote")

    # Upload all scripts
    print("\n--- Uploading experiment scripts ---")
    exp_to_run = EXPERIMENTS if which == 'all' else [e for e in EXPERIMENTS if which in e]
    if not exp_to_run:
        print(f"No experiments matching '{which}'")
        sys.exit(1)

    for exp_file in exp_to_run:
        local_path = os.path.join(LOCAL_DIR, exp_file)
        remote_path = REMOTE_DIR + '/' + exp_file  # Force Unix path
        print(f"  Uploading {exp_file}...")
        sftp.put(local_path, remote_path)

    print(f"\nUploaded {len(exp_to_run)} scripts to {REMOTE_DIR}")

    # Install dataset dependencies if needed
    print("\n--- Checking dependencies ---")
    out, err, code = run_command(ssh,
        "/root/miniconda3/bin/pip list 2>/dev/null | grep -iE 'datasets|evaluate' || echo 'need_install'")
    if 'need_install' in out:
        print("  Installing datasets...")
        run_command(ssh, "/root/miniconda3/bin/pip install datasets evaluate -q", timeout=120)

    # Run experiments
    print("\n" + "=" * 70)
    print("Running Experiments")
    print("=" * 70)

    results_summary = []

    for exp_file in exp_to_run:
        exp_name = exp_file.replace('.py', '')
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"Start: {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        remote_path = os.path.join(REMOTE_DIR, exp_file)
        cmd = f"cd {REMOTE_DIR} && /root/miniconda3/bin/python {exp_file} 2>&1"
        out, err, exit_code = run_command(ssh, cmd, timeout=3600)

        # Print without special chars that break GBK encoding
        safe_out = out.encode('ascii', errors='replace').decode('ascii')
        print(safe_out[-2000:] if len(safe_out) > 2000 else safe_out)
        if err:
            print(f"STDERR (last 500 chars):\n{err[-500:]}")

        status = "OK" if exit_code == 0 else f"FAILED (exit={exit_code})"
        print(f"\n{exp_name}: {status}")

        results_summary.append({
            'experiment': exp_name,
            'status': status,
            'exit_code': exit_code,
            'output_last_500': out[-500:] if out else '',
            'stderr_last_200': err[-200:] if err else '',
        })

    # Summary
    print("\n" + "=" * 70)
    print("All Experiments Complete")
    print("=" * 70)
    for r in results_summary:
        symbol = "[OK]" if r['exit_code'] == 0 else "[FAIL]"
        print(f"  {symbol} {r['experiment']}: {r['status']}")

    # List result files
    out, err, code = run_command(ssh, f"ls -la {RESULT_DIR}/")
    print(f"\nResult files:\n{out}")

    # Save local summary
    summary_path = os.path.join(LOCAL_DIR, 'experiment_status.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiments': results_summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"Local summary saved to {summary_path}")

    # Download results
    print("\n--- Downloading results ---")
    local_results_dir = os.path.join(LOCAL_DIR, 'results')
    os.makedirs(local_results_dir, exist_ok=True)

    # List remote result files
    out, err, code = run_command(ssh, f"ls {RESULT_DIR}/")
    remote_files = [f.strip() for f in out.strip().split('\n') if f.strip().endswith('.json')]

    for rf in remote_files:
        remote_path = RESULT_DIR + '/' + rf
        local_path = local_results_dir + '\\' + rf
        print(f"  Downloading {rf}...")
        try:
            sftp.get(remote_path, local_path)
        except Exception as e:
            print(f"    Failed: {e}")

    print(f"\nDownloaded results to {local_results_dir}")

    # Cleanup
    sftp.close()
    ssh.close()
    print("\nDone. SSH connection closed.")

if __name__ == '__main__':
    main()
