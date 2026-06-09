#!/bin/bash
# Download Qwen2.5-7B and WikiText-2 via HF mirror
set -e

export HF_ENDPOINT=https://hf-mirror.com

echo "=== Downloading Qwen2.5-7B ==="
/root/miniconda3/bin/python << 'PYEOF'
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Downloading Qwen2.5-7B...")
try:
    path = snapshot_download(
        'Qwen/Qwen2.5-7B',
        cache_dir='/root/autodl-tmp',
        resume_download=True
    )
    print(f"Qwen2.5-7B downloaded to: {path}")
except Exception as e:
    print(f"Failed: {e}")
    # Try alternative
    try:
        path = snapshot_download(
            'Qwen/Qwen2.5-7B-Instruct',
            cache_dir='/root/autodl-tmp',
            resume_download=True
        )
        print(f"Qwen2.5-7B-Instruct downloaded to: {path}")
    except Exception as e2:
        print(f"Also failed: {e2}")
PYEOF

echo ""
echo "=== Downloading WikiText-2 ==="
/root/miniconda3/bin/pip install datasets -q 2>&1 | tail -1
/root/miniconda3/bin/python << 'PYEOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

print("Downloading WikiText-2...")
try:
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wiki.save_to_disk('/root/autodl-tmp/wikitext2')
    print(f"WikiText-2 downloaded: {len(wiki)} samples")
except Exception as e:
    print(f"Failed: {e}")
    # Try C4
    print("Trying C4 (realnewslike)...")
    try:
        c4 = load_dataset('c4', 'realnewslike', split='validation', streaming=True)
        # Save first 1000 samples
        samples = []
        for i, s in enumerate(c4):
            if i >= 1000: break
            samples.append(s)
        import json
        with open('/root/autodl-tmp/c4_samples.json', 'w') as f:
            json.dump(samples, f)
        print(f"C4: saved {len(samples)} samples")
    except Exception as e2:
        print(f"C4 also failed: {e2}")
PYEOF

echo "--- Done ---"
ls -la /root/autodl-tmp/ | grep -E 'qwen|wiki|c4|Qwen'
