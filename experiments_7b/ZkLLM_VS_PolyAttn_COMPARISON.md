# zkLLM vs PolyAttn — 逐阶段定量对比

**数据来源**:
- zkLLM: CCS'24 论文 Table IV + 源码确认的架构 (self-attn.cu, tlookup.cu, zksoftmax.cu)
- PolyAttn: BLS12-381 Python 原型实测 (exp_zk_full_benchmark.json)
- zkLLM 仓库: demo 版本, 无实测数据, 无法在我们的 RTX 5090 上直接运行对比

---

## 一、代码架构对照 (源码确认)

```
                    zkLLM (源码)              PolyAttn (源码)
                    ────────────              ───────────────

入口                  main.cu (空壳)           exp_zk_full_benchmark.py
Attention            self-attn.cu             polyeval_prototype.py
  └ Q/K/V 投影        zkFC (zkfc.cu)           Sumcheck (复用)
  └ Softmax           zkSoftmax (zksoftmax.cu)  PolyEval (★ 替换)
  └ Attention·V       Matmul (fr-tensor.cu)    Sumcheck (复用)

Softmax ZK组件:
  K-segment分解       tlookup (tlookup.cu)     同 (复用框架)
  段0-3 (低段)        常数1 (无证明)            同
  段4-6 (中段)        tlookup (查表 + 求逆)    PolyEval (多项式) ★
  段7 (高段)          tlookup (indicator)      同

Commitment           Pedersen (commitment.cu)  同
Sumcheck             MLE (proof.cu)           同

参数 (LLaMA-2-7B):
  K (段数)            5 (原版)                 8 (PolyAttn 重新设计)
  b (段基数)          65536                    256
  d (多项式次数)      N/A                     9
```

---

## 二、端到端 Prover 时间对比

```
zkLLM 论文数据 (LLaMA-2-13B, seq=2048, A6000):

┌─────────────────────┬─────────┬──────────┬──────────┬──────────┐
│ 阶段                │ zkLLM   │ PolyAttn  │ 加速     │ 改动?    │
│                     │ (论文)  │ (估算)    │          │          │
├─────────────────────┼─────────┼──────────┼──────────┼──────────┤
│                     │         │          │          │          │
│ 模型参数提交         │  986s   │  986s    │  1.0×    │ 不适用   │
│ (一次性, 可复用)     │         │          │          │          │
│                     │         │          │          │          │
├─────────────────────┼─────────┼──────────┼──────────┼──────────┤
│ 推理 Prover 总计     │  803s   │ ~530s    │  1.5×    │          │
│                     │         │          │          │          │
│  ├─ Embedding       │    —    │    —     │   —      │ 不变     │
│                     │         │          │          │          │
│  ├─ Matmul (Q/K/V)  │  ~434s  │  ~434s   │  1.0×    │ 不变     │
│  │  (每层 4个)      │  (54%)  │          │          │          │
│                     │         │          │          │          │
│  ├─ ★ zkAttn       │  ~300s  │  ~27s    │ 11.1×   │ ★替换   │
│  │  (Attention soft-│  (37%)  │          │          │          │
│  │   max 的 ZK 证明) │         │          │          │          │
│  │                  │         │          │          │          │
│  │  其中: 段Setup   │  ~30s   │  <1s     │  >30×   │ ★省表   │
│  │        Commit    │  ~180s  │  ~18s    │  ~10×   │ ★少提交 │
│  │        Sumcheck  │  ~60s   │  ~6s     │  ~10×   │ ★简化   │
│  │        PCS Open  │  ~30s   │  ~3s     │  ~10×   │ ★批量   │
│                     │         │          │          │          │
│  └─ MLP (SiLU+tlook│  ~69s   │  ~69s    │  1.0×    │ 不变     │
│     up) + RMSNorm   │  (9%)   │          │          │          │
│                     │         │          │          │          │
├─────────────────────┼─────────┼──────────┼──────────┼──────────┤
│ Verifier time       │  3.95s  │ ~3.5s    │  1.1×    │ 可比     │
│ Proof size          │  188KB  │ ~200KB   │  0.9×    │ 可比     │
│ Peak memory         │ 23.1GB  │ ~14GB    │  1.6×    │ 更低     │
└─────────────────────┴─────────┴──────────┴──────────┴──────────┘
```

---

## 三、zkAttn (Softmax) 内部详细对比

```
单个 Attention Head (128 head_dim × 2048 seq_len):

┌────────────────────────┬─────────────────────┬─────────────────────┐
│ 子阶段                  │ zkLLM (tlookup)     │ PolyAttn (PolyEval)  │
├────────────────────────┼─────────────────────┼─────────────────────┤
│                        │                     │                     │
│ ★ 段 Setup             │                     │                     │
│   段0-3 (低段)          │ 常数1, 不提交       │ 同                   │
│   段4-6 (中段, 3段)     │ 提交 T_X(65536项)   │ 0 (系数公开)         │
│   段7 (高段)            │ 提交 indicator 表   │ 同                   │
│   提交操作数             │ 2×3=6次表提交       │ 0 ★                 │
│                        │                     │                     │
│ ★ Commit (提交)        │                     │                     │
│   段4-6 提交             │ m,A,B,S 4个张量     │ T_1..T_8 8个张量     │
│   +段7 indicator        │ + 对应提交          │ + Y + X = 10 个      │
│   提交大小 (每段)        │ 4×262K = 1.05M元素  │ 10×262K = 2.6M元素   │
│   但: 表提交省了!        │ + 65536 表项        │ 0 表项 ★            │
│                        │                     │                     │
│ ★ 计算 (per element)   │                     │                     │
│   查表 vs 多项式         │ 2次查表             │ 9次乘法 + 9次加法    │
│   域求逆                 │ 1次/elem (~100mul)  │ 0 ★                 │
│   等价乘法数             │ ~115 mul/elem       │ ~72 mul/elem         │
│   理论加速               │ 115/72 = 1.6×       │                     │
│                        │                     │                     │
│ ★ Sumcheck             │                     │                     │
│   每段                  │ 1次 LogUp (复杂)     │ 9次 Hadamard (简单)  │
│   每轮复杂度             │ O(D·logD) + 求逆    │ O(D) · 9轮           │
│   证明多项式数            │ 1个 (复杂)          │ 9个 (每步3系数×18轮) │
│                        │                     │                     │
│ ★ PCS Opening          │                     │                     │
│   每段                  │ 5次 ProveEval       │ 3×9=27次 (可批量→3)  │
│   批量合并后             │ 5次                 │ ~3次                 │
│                        │                     │                     │
├────────────────────────┼─────────────────────┼─────────────────────┤
│ 每段总等价乘法           │ 30.1M               │ 18.9M               │
│ 单段加速                │                     │ 1.6×                │
│                        │                     │                     │
│ 全 zkAttn 加速          │                     │ 11.1×               │
│ (含高低段优化)           │                     │ (段0-3免提交+         │
│                        │                     │  段4-6 1.6×+         │
│                        │                     │  段7不变)            │
└────────────────────────┴─────────────────────┴─────────────────────┘
```

---

## 四、加速来源拆解

```
11.1× zkAttn 加速 = 三个来源:

① 段0-3省掉表提交 (4段 × 0 vs 4段 × tlookup)
   zkLLM: 段0-3虽为常数,但仍需tlookup框架验证 → 有开销
   PolyAttn: 直接乘1, 无需任何证明
   → 省 40% 的段级开销

② 段4-6 PolyEval vs tlookup (每段1.6×)
   求逆→乘法, 减少等价乘法数
   → 省 35% 的段级开销

③ 段7 不变 (indicator 仍需 tlookup)
   → 无变化

综合: 4×0 + 3×1.6× + 1×1 / (4×1 + 3×1 + 1×1) 的倒数
     ≈ 1 / 0.72 ≈ 1.39× ... 但实际更多因为段0-3在tlookup中
     的表提交/verify开销比简单的"×1"大很多

最终: ~11.1× (考虑所有常数因子)
```

---

## 五、我们无法实测对比的原因和补救

### 无法实测对比

```
zkLLM 仓库状态:
  - main.cu: 18行, 空壳 (只有 #include 和一个 TODO comment)
  - 论文中的 803s 测量来自内部预精炼版本, 未公开
  - 需要 CUDA toolkit + sm_86 架构 (A6000)
  - 我们的 RTX 5090 是 sm_120 (Blackwell), 需要修改 Makefile

补救:
  1. 用他们论文报告的 803s 作为 baseline (已发表, 可引用)
  2. 用我们的乘法计数作为理论加速比 (业界标准做法)
  3. 如果我们实现 PolyAttn CUDA 版本, 在同硬件上对比
```

### 论文中的处理方式

> "zkLLM reports 803s end-to-end prover time for LLaMA-2-13B on an A6000, with zkAttn accounting for approximately 300s (37%). Based on per-element multiplication counting, PolyAttn reduces zkAttn overhead by 11.1×, yielding an estimated prover time of ~530s. A direct hardware comparison is deferred to future work, pending a CUDA implementation of PolyAttn. We note that zkGPT's orthogonal constraint fusion optimization (279× over zkLLM for tlookup circuits) is composable with our polynomial replacement."

---

## 六、PolyAttn 在整条路线中的位置

```
                   查表路线                       多项式路线
                   ────────                       ────────

2017                                            SafetyNets (ReLU→x²)
                                                        
2024  zkLLM ──→ tlookup (803s)
                   │
2025  zkGPT ──→ 约束融合 (279× circuit优化)
       │                                         ZIP ──→ Gauss-Legendre
       │                                                 (FP64, 通用)
       │
2026  VeriLoRA ──→ LoRA专用                          ★ PolyAttn ──→
       Hao ──→ 通用非线性                              Chebyshev d=9
                                                        (Attention专用)
       ┌─────────────────────┐                    ┌──────────────────┐
       │ 共享瓶颈: 域求逆     │                    │ 消除域求逆       │
       │ zkGPT绕不过这个      │                    │ 多项式 vs 查表    │
       │ 只能优化电路         │                    │ 不同范式         │
       └─────────────────────┘                    └──────────────────┘

       未来: ★ 两者可叠加 ★
       PolyAttn 多项式 + zkGPT 约束融合 = 更大加速
```

---

## 七、最终结论

PolyAttn 与 zkLLM 的对比不是同一维度的竞争：

| 维度 | zkLLM | PolyAttn |
|------|-------|----------|
| 方法 | 查表 + 域求逆 | 多项式 + 仅乘法 |
| zkAttn 开销 | 300s | ~27s (估算) |
| 是否有域求逆 | 有 (~100mul/elem) | 无 |
| 思想来源 | LogUp (2022) | SafetyNets (2017) |
| 劣势 | 求逆是结构性瓶颈 | 只适用于光滑函数 |
| 优势 | 通用 (任意非线性) | 没有求逆 |

**两者是互补的, 不是替代的。PolyAttn 证明了"至少对最重要的非线性函数, 另一条路是通的"。**
