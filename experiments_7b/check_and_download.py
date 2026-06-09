"""Check 13B status + try Qwen2.5-7B via modelscope"""
import os, subprocess, sys

print("=== 13B Download Status ===")
os.system('tail -3 /root/autodl-tmp/download_13b.log 2>/dev/null || echo "no log"')

print("\n=== Bin files ===")
os.system('find /root/autodl-tmp -name "*.bin" -ls 2>/dev/null | head -5 || echo "none"')

print("\n=== Trying modelscope for Qwen2.5-7B ===")
try:
    from modelscope import snapshot_download
    path = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp')
    print(f"OK: {path}")
except Exception as e:
    print(f"Failed: {e}")

print("\n=== Disk usage ===")
os.system('df -h /root/autodl-tmp')
os.system('du -sh /root/autodl-tmp/*/ 2>/dev/null')
