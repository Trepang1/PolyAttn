"""Domain sweep for Chinese-Llama-2-7b PolyAttn PPL test."""
import torch, torch.nn.functional as F, numpy as np, time, sys
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_polynomial(lambda_val, M, d):
    n_cheb = max(d * 12, 64)
    k = np.arange(n_cheb)
    t_nodes = np.cos(np.pi * (k + 0.5) / n_cheb)
    x_nodes = (M / 2.0) * (t_nodes + 1.0)
    f_vals = np.exp(-lambda_val * x_nodes)
    cheb = np.zeros(n_cheb)
    for k_idx in range(n_cheb):
        cheb[k_idx] = (2.0/n_cheb)*np.sum(f_vals*np.cos(k_idx*np.pi*(k+0.5)/n_cheb))
    cheb[0] /= 2.0; cheb = cheb[:d+1]
    T = np.zeros((d+1,d+1)); T[0,0]=1.0
    if d>=1: T[1,1]=1.0
    for k_idx in range(2,d+1):
        T[k_idx,1:]+=2.0*T[k_idx-1,:-1]; T[k_idx,:]-=T[k_idx-2,:]
    mono=np.zeros(d+1)
    for k_idx in range(d+1):
        for j in range(k_idx+1):
            t_kj=T[k_idx,j]
            if abs(t_kj)<1e-16: continue
            for r in range(j+1):
                mono[r]+=t_kj*cheb[k_idx]*comb(j,r)*(2.0/M)**r*(-1.0)**(j-r)
    return list(mono)

def check_approx(coeffs, lam, M):
    grid=np.linspace(0,M,5001)
    return float(np.max(np.abs(np.exp(-lam*grid)-np.polyval(coeffs[::-1],grid))))

def horner_torch(c, x):
    y=torch.full_like(x,c[-1])
    for ci in reversed(c[:-1]): y=y*x+ci
    return y

def make_poly_softmax(M_domain, d=9):
    lam=1.0; coeffs=generate_polynomial(lam,M_domain,d)
    linf=check_approx(coeffs,lam,M_domain)
    cn=np.array(coeffs,dtype=np.float64)
    def fn(logits,dim=-1,dtype=None,**kw):
        x=logits.float(); mx=x.max(dim=dim,keepdim=True).values
        xs=x-mx; xc=torch.clamp(xs,-M_domain,0.0)
        y=horner_torch(cn,-xc); y=torch.clamp(y,min=1e-30)
        out=y/y.sum(dim=dim,keepdim=True)
        t=dtype if dtype is not None else logits.dtype
        return out.to(t) if t!=torch.float32 else out
    return fn,linf

@torch.no_grad()
def compute_ppl(model,tokenizer,texts,max_length=128,device='cuda'):
    model.eval(); tl,tt=0.0,0
    for text in texts:
        enc=tokenizer(text,return_tensors='pt',truncation=True,max_length=max_length)
        ids=enc['input_ids'].to(device)
        if ids.size(1)<4: continue
        try:
            out=model(ids,labels=ids)
            if out.loss is not None and not torch.isnan(out.loss):
                tl+=out.loss.item()*ids.size(1); tt+=ids.size(1)
        except: pass
    if tt==0: return float('inf'),float('inf')
    a=tl/tt; return float(np.exp(a)),float(a)

MODEL='/root/autodl-tmp/chinese-llama-2-7b'
DEVICE='cuda'

texts=[
    "人工智能技术正在快速发展，大语言模型成为了当前最受关注的研究方向之一。",
    "深度学习的发展推动了自然语言处理领域的巨大进步。",
    "在当今数字化时代，数据安全和隐私保护成为了越来越重要的研究课题。",
    "计算机视觉技术使得机器能够理解和分析图像内容。",
    "量子计算作为一种新型计算范式，有望在密码学等领域带来革命性变化。",
    "中国在人工智能领域的研究投入不断增加，推动了技术创新。",
    "自然语言处理是人工智能的重要分支，涵盖了多个方向。",
    "大数据时代，如何有效保护用户隐私成为了关键挑战。",
]*5

tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL,torch_dtype=torch.bfloat16,device_map='auto',
    trust_remote_code=True,attn_implementation='eager')
model.eval()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB')

orig_softmax=F.softmax
ppl_orig,loss_orig=compute_ppl(model,tokenizer,texts,128,DEVICE)
print(f'Original PPL: {ppl_orig:.4f} (loss={loss_orig:.4f})\n')

print(f'{"Domain":<12} {"Linf":<12} {"PPL":<12} {"Delta":<12} {"Change%":<10}')
print('-'*58)

domains=[4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0,15.0,20.0]
for dm in domains:
    poly_fn,linf=make_poly_softmax(dm,d=9)
    F.softmax=poly_fn
    t0=time.time()
    ppl,loss=compute_ppl(model,tokenizer,texts,128,DEVICE)
    dt=time.time()-t0
    delta=ppl-ppl_orig
    print(f'[-{dm:<4.0f},0]     {linf:<12.2e} {ppl:<12.4f} {delta:<+12.4f} {delta/ppl_orig*100:<+10.2f}%  ({dt:.0f}s)')

F.softmax=orig_softmax

# Best
best_dm=min(domains,key=lambda dm: abs(make_poly_softmax(dm,d=9)[0] and compute_ppl(model,tokenizer,texts,128,DEVICE)[0]-ppl_orig))
print(f'\nDone. Original PPL={ppl_orig:.4f}')
print('Recommendation: domain [-8,0] or [-10,0] for best fidelity/efficiency trade-off.')
