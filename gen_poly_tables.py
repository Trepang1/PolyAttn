import numpy as np, os, struct
from math import comb

SCALE = 1 << 16

def cheb_coeffs(lam, M, d):
    n = max(d*12,64); k = np.arange(n)
    tn = np.cos(np.pi*(k+0.5)/n); xn = (M/2)*(tn+1)
    fv = np.exp(-lam*xn)
    c = np.zeros(n)
    for ki in range(n): c[ki] = (2.0/n)*np.sum(fv*np.cos(ki*np.pi*(k+0.5)/n))
    c[0]/=2.0; c=c[:d+1]
    T=np.zeros((d+1,d+1)); T[0,0]=1.0
    if d>=1: T[1,1]=1.0
    for ki in range(2,d+1): T[ki,1:]+=2.0*T[ki-1,:-1]; T[ki,:]-=T[ki-2,:]
    mono=np.zeros(d+1)
    for ki in range(d+1):
        for j in range(ki+1):
            if abs(T[ki,j])<1e-16: continue
            for r in range(j+1): mono[r]+=T[ki,j]*c[ki]*comb(j,r)*(2.0/M)**r*(-1.0)**(j-r)
    return list(mono)

def poly_val(coeffs, x):
    d=len(coeffs)-1; y=coeffs[d]
    for c in reversed(coeffs[:d]): y=y*x+c
    return y

# Match zkLLM self-attn parameters
bs = [256, 1048576, 1048576]  # base sizes
scaling_factor = 1 << 32  # 2^32
d = 9

WORK = "/root/autodl-tmp/zkllm-workdir/Llama-2-7b"

for seg_idx, b in enumerate(bs):
    lam = b / scaling_factor
    M = b - 1  # domain [0, b-1]
    print(f"Segment {seg_idx}: b={b}, lam={lam:.6f}, lam*M={lam*M:.3f}")
    
    coeffs = cheb_coeffs(lam, M, d)
    # Generate table: theta_k * p(x) for x in [0, b-1]
    # theta_k is absorbed into the polynomial; for demo, use theta=1
    # In actual zkLLM, theta_k = thetas[seg_idx] (2^18 or 2^22)
    table = []
    for x in range(b):
        pv = poly_val(coeffs, x)  # p(x) approx exp(-lam*x)
        pv = max(pv, 0.0)  # clamp negative
        # Encode as 64-bit integer split into 2 x uint32 (matching zkLLM format)
        val64 = int(pv * SCALE)  # scale to match F_p encoding
        lo = val64 & 0xFFFFFFFF
        hi = (val64 >> 32) & 0xFFFFFFFF
        # Fr_t format: {lo, hi, 0, 0, 0, 0, 0, 0}
        table.extend([lo, hi, 0, 0, 0, 0, 0, 0])
    
    path = f"{WORK}/polyeval_table_{seg_idx}.bin"
    with open(path, 'wb') as f:
        for v in table:
            f.write(struct.pack('I', v))
    size_mb = os.path.getsize(path)/1e6
    print(f"  Saved {path} ({size_mb:.1f}MB)")
    
print("PolyAttn tables generated!")
