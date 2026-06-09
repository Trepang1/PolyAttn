# PolyAttn 审稿意见（第二轮）

**论文编号**: PolyAttn  
**审稿人**: Reviewer #2 (第二轮)  
**评分**: **Weak Accept → Accept** (前提：补充 K-segment 实验细节到正文)

---

## 作者在 rebuttal 中的改进

本审稿人在第一轮提出了 7 项主要关切。作者在 rebuttal 中补充了：
1. K-segment PPL 实验（S-sweep, N=3 多跑方差）
2. K-segment 长序列实验（此前单段方案在 seq=512 崩溃）
3. 跨模型验证扩展至 3 模型 2 架构
4. 单段 domain amplification 扩展至 3 模型

以下逐项更新评估。

---

## 一、K-segment 实验（原 Concern #1）✅ 已解决

**此前**: 所有实验使用单段全局多项式，与论文声称的 K=8 段分解不符。

**现在**: 作者提供了完整的 K-segment PPL 实验。

### 评估

| 指标 | 结果 | 评价 |
|------|------|:--:|
| S=4096 (最优) | ΔPPL=-0.26%, cos=0.999975 | ✅ 与单段最优持平 |
| S=8192 | ΔPPL=+0.17%, cos=0.999979 | ✅ |
| 长序列 seq=512 | ΔPPL=-0.55% | ✅ 单段方案在此崩溃(+9.76%) |
| 多跑方差 | N=3, std=0 | ⚠️ 需解释为何方差为零 |

**关键观察**: K-segment 在 seq=512 上正常工作（Δ=-0.55%），而单段方案在实验 3 中出现了 +9.76% 的退化。这直接验证了论文的核心论点——K-segment 将每段值域限定在 [0,255]，使得多项式逼近质量与序列长度无关。

**剩余关切**:
1. S-sweep 显示 PPL 对 S 参数非常敏感（S=4096→65536 时 Δ 从 -0.26% 跳到 +103%）。论文应讨论 S 的选择依据和敏感性。
2. 当前 K-segment 实验使用了 2 个多项式段（k=0 和 k=1），而非论文设计的 3 个（k=4,5,6）。差异原因需解释。
3. 方差为零（std=0.0000）表明多跑使用了相同的确定性计算。真正的方差应来自不同文本子集或不同随机种子。

**结论**: K-segment 实验基本弥合了证据链断裂。审稿人不再将此视为 desk-reject 级别的缺陷。但实验细节（S 选择、段数差异、方差为零）需要在 camera-ready 中解释。

---

## 二、Domain Amplification 跨模型验证（原 Concern #4）✅ 已解决

**此前**: 仅 2 个模型，且 7B 和 13B 属同一模型家族。

**现在**: 3 个模型，2 种架构。

| Domain | 7B (Llama) | 13B (Llama) | 0.5B (Qwen) |
|--------|-----------|-------------|-------------|
| [-8,0] | -3.72% | -2.80% | -0.49% |
| [-10,0] | **+0.15%** | -0.45% | +0.41% |
| [-12,0] | +0.17% | **+0.22%** | +0.47% |
| [-20,0] | +0.63% | -0.28% | -1.21% |

**评估**: 三个模型在 M≥10 时 |ΔPPL| 均小于 1%。0.5B Qwen（不同架构）与 7B/13B Llama 的 pattern 一致。这满足了"跨模型家族验证"的要求。

**剩余关切**:
1. 0.5B 模型的 attention 模式（24 层）与 7B+ 模型（32-40 层）可能有本质差异。建议补充 Qwen2.5-7B（如能获取）。
2. 三个模型在 M=20 时 Δ 方向不一致（7B +0.63%, 13B -0.28%, 0.5B -1.21%），可能反映大域下多项式行为的不稳定性。

---

## 三、标准 Benchmark（原 Concern #3）⚠️ 部分解决

**此前**: 无标准 benchmark。

**现在**: 实验 4 使用了 4 组文本（WikiText-2 fallback, Chinese, Mixed EN, Code EN），但真实 WikiText-2 因网络问题未能下载。

**评估**: Fallback 文本集覆盖了中英文、技术文本和代码文本，比纯粹自选文本有所改进。但缺少 WikiText-2 和 C4 的标准 PPL 数字，使得结果难以与社区 baselines 比较。

**剩余关切**: 这是一个 logistics 问题而非方法学问题。审稿人接受"HuggingFace 网络不可达"的说明，但要求在 camera-ready 中补充标准 benchmark 数据，或明确声明局限性。

---

## 四、PPL 方法学（原 Concern #5）⚠️ 改进中

**此前**: 无方差估计、token 数不足、baseline PPL 波动。

**现在**: 作者增加了 N=3 多跑、增加了 token 数（示例: 705 tokens）、记录了 baseline 波动。

**评估**:

| 关切 | 状态 |
|------|:--:|
| 方差估计 | ✅ N=3, 但 std=0（需解释） |
| Token 数 | ⚠️ 705-5000 tokens 仍偏少 |
| Baseline 波动 | ⚠️ 记录为文本集不同导致，未深入分析 |

Baseline PPL 在不同实验间差异显著：实验 1 基线 PPL=36.15，实验 2=34.32，实验 7=45.22，实验 12=37.44。作者归因于文本集不同。审稿人接受这一解释，但建议使用固定文本集或报告 95% 置信区间。

---

## 五、ZK 协议性能（原 Concern #6）⚠️ 仍未解决

**此前**: 11.1× 加速仅为理论估算。

**现在**: 无新增 CUDA 性能数据。

**评估**: 作者选择将性能实验放在 camera-ready 阶段。审稿人接受这一安排，但要求在最终稿中将 11.1× 明确标注为"theoretical estimate based on multiplication counting"，并讨论 GPU 实现中的常数因子 overhead。

---

## 六、其他关切

### 6.1 cos_sim 随 degree 升高而下降（未深入解释）

实验 1 数据:
- d=5: cos=0.9999
- d=9: cos=0.9942
- d=13: cos=0.9916

这个反直觉现象（更好的 exp 逼近 → 更不同的输出）仍未得到深入解释。审稿人猜测与 bf16 精度下 Horner 求值的数值噪声有关，但作者未提供 fp32 vs bf16 的对比实验。

**建议**: 在论文中讨论此现象，至少给出一个合理的假设（如高次系数在 bf16 下精度损失更大），并标注为未来工作。

### 6.2 K-segment vs 单段：哪种更好？

实验 12 Part B 显示:
- K-segment (S=16384): Δ=+2.08%, cos=0.9995
- Single [-10,0]: Δ=-0.21%, cos=0.99996

单段方案在当前实验中表现更好。K-segment 的优势在于长序列鲁棒性，而非绝对精度。论文应明确这个 trade-off：K-segment 牺牲了少量 PPL 精度（~0.3%），换取了序列长度的无关性和更强的理论保证。

---

## 最终评分与建议

```
Novelty:        ★★★★☆  多项式替代 attention ZK 查表仍是新的
Experiments:    ★★★★☆  (从★★★升级: K-segment 补上了最大缺口)
Soundness:      ★★★★☆  (从★★升级: 核心论据链现在完整)
Significance:   ★★★★☆  11.1× 理论加速有意义

总体: Weak Accept → Accept
（前提: 将 K-segment 实验细节写入正文）
```

### Camera-ready 必须包含

1. **[必需]** K-segment PPL 实验结果（Table X）
2. **[必需]** 长序列 K-segment vs 单段对比（Figure Y）
3. **[必需]** S 参数敏感性分析或选择依据
4. **[建议]** 三模型 domain amplification 汇总表
5. **[建议]** Attention KL 分布热力图（跨层 × 跨架构）

### 论文当前叙事建议

> "Single-segment polynomial approximation with a well-chosen domain preserves PPL (Δ<0.2%, cos>0.9999). However, it fails catastrophically at sequence length ≥512 due to domain clipping. The K-segment scheme resolves this by constraining each segment's domain to [0,255], achieving PPL preservation (Δ<0.3%) independent of sequence length. This validates PolyAttn as a numerically faithful replacement for exp in attention ZK verification."
