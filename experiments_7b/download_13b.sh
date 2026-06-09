#!/bin/bash
# Download Chinese-Llama-2-13b from ModelScope
set -e

/root/miniconda3/bin/pip install modelscope -q 2>&1 | tail -2

/root/miniconda3/bin/python << 'PYEOF'
from modelscope import snapshot_download
import os

print("Trying ModelScope download...")
try:
    model_dir = snapshot_download(
        'ChineseAlpacaGroup/chinese-llama-2-13b',
        cache_dir='/root/autodl-tmp'
    )
    print(f"Downloaded to: {model_dir}")
except Exception as e:
    print(f"ModelScope error: {e}")
    # Try HF mirror
    print("Trying HF mirror...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from huggingface_hub import snapshot_download as hf_snapshot
    try:
        model_dir = hf_snapshot('hfl/chinese-llama-2-13b', cache_dir='/root/autodl-tmp')
        print(f"Downloaded to: {model_dir}")
    except Exception as e2:
        print(f"HF mirror error: {e2}")
PYEOF

echo "--- Done ---"
ls -la /root/autodl-tmp/
