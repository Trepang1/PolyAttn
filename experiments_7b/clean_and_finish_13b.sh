#!/bin/bash
set -e

echo "=== Before cleanup ==="
df -h /root/autodl-tmp | tail -1

echo "Cleaning temp files..."
rm -rf /root/autodl-tmp/._____temp
rm -rf /root/autodl-tmp/models--hfl--chinese-llama-2-13b
rm -f /root/autodl-tmp/download_*.log
rm -f /root/autodl-tmp/auto_*.log

echo "=== After cleanup ==="
df -h /root/autodl-tmp | tail -1
echo ""
echo "=== 13B shards ==="
ls -la /root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b/*.bin 2>/dev/null || echo "no bins yet"

echo ""
echo "=== Downloading shard 3 via HF mirror ==="
export HF_ENDPOINT=https://hf-mirror.com
/root/miniconda3/bin/python << 'PYEOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

print("Downloading shard 3...")
path = hf_hub_download(
    repo_id='hfl/chinese-llama-2-13b',
    filename='pytorch_model-00003-of-00003.bin',
    cache_dir='/root/autodl-tmp',
    local_dir='/root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b',
    local_dir_use_symlinks=False
)
print(f"Downloaded to: {path}")
size = os.path.getsize(path)
print(f"Size: {size/1e9:.2f} GB")
PYEOF

echo ""
echo "=== Final check ==="
ls -la /root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b/*.bin
df -h /root/autodl-tmp | tail -1

echo ""
echo "=== Running 13B experiment ==="
/root/miniconda3/bin/python /root/autodl-tmp/experiments_7b/exp10_13b_domain.py 2>&1
