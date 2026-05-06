# Stanford Bunny Multimodal Experiment Framework

## 核心设计思路

### 两个真实的独立模态

之前的版本把 latent = tanh(coords @ W) 当作第二模态，但这不是真正的多模态——latent 只是同一份 3D 数据的线性投影，并不携带任何独立信息。

新设计引入了两个真实独立的数据来源：

| 模态 | 内容 | 随时间变化 |
|------|------|-----------|
| **Modality A** (几何) | 3D 坐标 — 由 ODE 驱动演化 | ✓ 随系统演化 |
| **Modality B** (语义) | Bunny 部位标签：`body=0, left_ear=1, right_ear=2` | ✗ 固定不变 |

部位标签在数据生成时确定，与任何时刻的 3D 坐标完全无关：

```
body     : 前 2/3 个点 → 球面均匀分布（golden-angle）
left_ear : 后 1/6 个点 → x = -0.25 的竖向圆柱
right_ear: 后 1/6 个点 → x = +0.25 的竖向圆柱
```

---

### 对齐分数：Fisher 组间散度比

```
semantic_alignment_score(vectors, part_labels)
  = between_class_scatter / total_scatter  ∈ [0, 1]

0 → 三个部位的点在该向量空间里完全混杂
1 → 三个部位的点形成完全分离的聚类
```

每个实验追踪三条曲线：

```
spatial_score(t)  = semantic_alignment_score(coords_t,         part_labels)
                    → 3D 坐标里各部位分离程度

tracking_score(t) = semantic_alignment_score(latent(coords_t), part_labels)
                    → latent 空间里各部位分离程度

static_score      = semantic_alignment_score(latent(coords_0), part_labels)
                    → 初始 latent 的基线（常量参考线）
```

**latent_amplification = tracking_score(t) − spatial_score(t)**
> 正值：latent 编码放大了几何中已有的部位结构
> 负值：latent 编码压缩/丢失了部位结构

---

## 四个实验的科学假设

### Diffusion Multimodal

**系统行为**：Bunny 100 个粒子在 ProbabilityFlowODE 驱动下，被拉向 4 个 GMM 聚类中心。

**假设**：4 个 GMM 中心并不对应 3 个部位，因此 spatial_score 不会系统性上升。扩散过程在几何上是"破坏性"的，部位结构可能被打散。

**实验结果（参考）**：
- 3D 粒子全部收敛到 4 个簇（purity: 0% → 100%）
- spatial_score 保持平稳甚至下降（GMM 中心与部位边界不对应）
- latent_amplification < 0（latent 没有额外保留部位信息）

---

### Manifold Multimodal（t-SNE）

**系统行为**：Bunny 100 个点以 10D（3D + 7D 噪声）作为高维输入，t-SNE ODE 找到一个保留邻域结构的 3D 嵌入。

**假设**：t-SNE 保局部邻域，同一部位的点（body 内部、ear 内部）彼此相邻，因此 spatial_score 应从初始的随机散点（≈0）上升。

**实验结果（参考）**：
- KL divergence: 1.48 → 0.74（嵌入质量提升）
- spatial_score: 0.01 → 0.47（从随机散点到部位聚类）
- tracking_score: 0.01 → 0.41（latent 跟随几何聚类上升）
- t-SNE 从零开始重新发现了 Bunny 的部位结构

---

### CANDY Diffusion Multimodal（核心实验）

**系统行为**：`CANDYDiffusionSystem` 融合三种机制：
- **CANDY masking**：特征提取，`Y → tanh(Y @ Wp)` 保留输入结构
- **U-Net fusion**：`W_fuse`（编码器）+ `W_unet`（解码器）计算残差重建力
- **Graph schedule g(t)**：随时间从噪声特征驱动 → 目标 Ground Truth 引导

**多模态关键设计**：CANDY 目标类别设为三个 Bunny 部位的质心（而非随机中心），且每个粒子的目标 = 自己所属部位的质心：

```
target[0] (body)      = 身体点云的 3D 质心
target[1] (left_ear)  = 左耳点云的 3D 质心
target[2] (right_ear) = 右耳点云的 3D 质心
```

**假设**：CANDY 同时通过 U-Net 重建路径和 g(t) 引导把每个粒子拉向自己的部位质心，spatial_score 应从初始约 0.37 上升到接近 1.0。

**实验结果（参考）**：
- CANDY 收敛误差：2.53 → 0.16（接近完全收敛）
- spatial_score：0.37 → **1.00**（部位完全分离）
- tracking_score：0.43 → **1.00**（latent 同步到达完美聚类）

**对比意义**：
| 实验 | CANDY 目标 | spatial 结果 |
|------|-----------|-------------|
| diffusion_multimodal | 4个随机GMM中心 | 平稳（≈0.37） |
| candy_multimodal     | 3个部位质心   | 0.37 → 1.00 |

这个对比直接说明：**目标对齐是多模态收敛的关键**，U-Net + g(t) 本身不产生部位分离，对齐的目标才是驱动力。

---

### Neural Multimodal（Hopfield）

**系统行为**：Bunny 100 个点初始化 Hopfield 神经元的前 3 维，网络驱动向 3 个吸引子模式演化。吸引子设定为三个部位的质心（body / left_ear / right_ear）。

**假设**：吸引子对应部位质心，若收敛成功，spatial_score 应上升。

**关键设计**：
```python
# 吸引子 = 部位质心（从 Bunny 数据自动计算）
patterns[0] = body_centroid      # ≈ (0, 0, 0)
patterns[1] = left_ear_centroid  # ≈ (-0.16, 0, 0.82)
patterns[2] = right_ear_centroid # ≈ (+0.16, 0, 0.82)
```

---

## 文件结构

```
multimodal_data.py
  ├── generate_part_labels(num_points)          → (N,) int   Modality B
  ├── semantic_alignment_score(vectors, labels) → float [0,1]
  └── MultimodalAlignment
        ├── encode_to_latent(coords_3d)          → (N, latent_dim)
        └── compute_alignment_score(latent, part_labels) → float [0,1]

experiments/
  ├── candy_multimodal_experiment.py      ← CANDY + U-Net + g(t)（核心）
  ├── diffusion_multimodal_experiment.py  ← ProbabilityFlowODE
  ├── manifold_multimodal_experiment.py   ← t-SNE ODE
  └── neural_multimodal_experiment.py     ← Hopfield Network
```

---

## 快速运行

```bash
cd experiments
python candy_multimodal_experiment.py      # 核心：CANDY + UNet，targets=部位质心
python diffusion_multimodal_experiment.py  # 对比：GMM targets ≠ 部位质心
python manifold_multimodal_experiment.py   # t-SNE 从随机散点发现部位结构
python neural_multimodal_experiment.py     # Hopfield attractors = 部位质心
```

每个实验输出：
- `results/*_metrics.png`  — spatial / tracking / static 三条对齐曲线
- `results/*_latent.png`   — 各时刻 3D 状态 + latent PCA（按部位着色）
- `results/*_results.json` — 所有指标

---

## 数学细节

### Fisher 组间散度比

$$\text{score} = \frac{B}{T}, \quad B = \sum_{c} n_c \| \mu_c - \mu \|^2, \quad T = \sum_i \| v_i - \mu \|^2$$

其中 $\mu_c$ 是类 $c$ 的质心，$\mu$ 是全局质心，$n_c$ 是类 $c$ 的点数。

- $B/T \to 0$：类间距离远小于总体分散度，部位没有聚类
- $B/T \to 1$：总体分散度几乎完全来自类间差异，各部位紧密聚类

### latent 编码

$$z_i = \tanh\!\left(\hat{x}_i \cdot W\right) \times 2.0, \quad W \in \mathbb{R}^{3 \times d}$$

其中 $\hat{x}_i$ 是归一化后的 3D 坐标，$W$ 在第一次 `encode_to_latent` 调用时随机初始化并固定。解码使用伪逆 $W^\dagger$。

---

## 数据集说明

Stanford Bunny 在本框架中以合成方式生成（无需下载 PLY 文件），使用黄金角球面采样 + 双耳圆柱近似兔形点云。若需使用真实扫描数据：

| 资源 | 链接 |
|------|------|
| 官方主页 | https://graphics.stanford.edu/data/3Dscanrep/bunny/ |
| 高分辨率 PLY | 见官方页面 bun_zipper.ply (69,451 顶点) |

真实 PLY 加载：
```python
from multimodal_data import StanfordBunnyDataset
dataset = StanfordBunnyDataset()
coords = dataset.load_from_ply("path/to/bun_zipper.ply")
```

---

## 参考文献

- Turk & Levoy, "Zippered Polygon Meshes from Range Images", SIGGRAPH 1994
- Baltrušaitis et al., "Multimodal Machine Learning: A Survey and Taxonomy", IEEE TPAMI 2018
- van der Maaten & Hinton, "Visualizing Data using t-SNE", JMLR 2008
