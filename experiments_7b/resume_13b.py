"""Resume 13B download and verify all 3 shards"""
from modelscope import snapshot_download
import os, glob, time

print("Resuming 13B download...")
path = snapshot_download(
    'ChineseAlpacaGroup/chinese-llama-2-13b',
    cache_dir='/root/autodl-tmp'
)
print(f"Path: {path}")

# List all bins
bins = glob.glob(f"{path}/**/*.bin", recursive=True)
if not bins:
    bins = glob.glob(f"/root/autodl-tmp/**/chinese*llama*13b*/**/*.bin", recursive=True)

print(f"\nFound {len(bins)} .bin files:")
total = 0
for b in sorted(bins):
    size = os.path.getsize(b)
    total += size
    print(f"  {os.path.basename(b)}: {size/1e9:.2f} GB (dir: {os.path.dirname(b)})")

print(f"\nTotal: {total/1e9:.2f} GB")

# Check if we have all 3 shards
if len(bins) >= 3 and total > 20e9:
    print("\n*** ALL 3 SHARDS DOWNLOADED! ***")

    # If bins are in temp dir, move them
    temp_dir = '/root/autodl-tmp/._____temp/ChineseAlpacaGroup/chinese-llama-2-13b'
    final_dir = '/root/autodl-tmp/ChineseAlpacaGroup/chinese-llama-2-13b'

    if os.path.exists(temp_dir):
        import shutil
        for f in os.listdir(temp_dir):
            src = os.path.join(temp_dir, f)
            dst = os.path.join(final_dir, f)
            if not os.path.exists(dst):
                print(f"Moving {f} to final dir...")
                shutil.move(src, dst)
        print("Moved temp files to final dir")

    print("Ready for experiment!")
else:
    print(f"\nStill incomplete: {len(bins)}/3 shards, {total/1e9:.2f}/25 GB")
