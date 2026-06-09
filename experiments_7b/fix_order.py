# Fix: set_poly_workdir BEFORE zkSoftmax constructor
path = "/root/autodl-tmp/zkllm-ccs2024/self-attn.cu"
lines = open(path).readlines()
# Find the two lines and swap them
for i in range(len(lines)-1):
    if "zkSoftmax softmax" in lines[i] and "set_poly_workdir" in lines[i+1]:
        lines[i], lines[i+1] = lines[i+1], lines[i]
        print(f"Swapped lines {i+1} <-> {i+2}")
        break
open(path, "w").writelines(lines)
# Verify
for i, l in enumerate(open(path).readlines()):
    if "set_poly" in l or "zkSoftmax softmax" in l:
        print(f"  Line {i+1}: {l.strip()}")
