# CSP PBMC3k 固定工作流程

## 一、当前固定方案

当前确认并固定的工作流程为：

```text
原始 PBMC3k counts
→ 固定质控、归一化及 2000 个高变基因选择
→ 构造 X_scaled
→ GibbsCSP（H=50，center=False）
→ RSP 对齐
→ 分批计算后验条件因子均值
→ 按对齐后载荷后验均值的列 L2 范数加权
→ 使用全部 50 列加权 score 构建欧氏邻域图
→ Leiden（resolution=0.40）
→ 9 个簇
→ 聚类图展示标签 1–9
→ Wilcoxon top-5 markers、UMAP、诊断和汇总报告
```

另设独立 PCA 参照流程：同一 `X_scaled` → 50 个主成分 → 前 40 个构建邻域图 → Leiden `resolution=0.7` → 8 簇。

PCA 仅作为参照，不视为细胞类型真值。ARI/AMI 只表示聚类一致性，不称为准确率。9 簇是当前固定表示和 Leiden 参数下的结果，不直接解释为真实细胞类型数或生物学最优。

本流程不再提供 `X_log`、`d10`、`d20`、pilot、可选 RSP、可选列数或其他敏感性实验分支。

## 二、数据预处理

实现文件：`PBMC/data_processing.py`。固定原始输入为 `PBMC/data/pbmc3k_raw.h5ad`，原始维度为 **2700 × 32738**。保存输入路径、SHA-256、软件版本和分阶段配置。

### 1. 过滤与质控

严格按以下顺序执行：

1. `sc.pp.filter_cells(adata, min_genes=200)`。
2. `sc.pp.filter_genes(adata, min_cells=3)`。
3. 以 `MT-` 前缀标记线粒体基因，调用 `sc.pp.calculate_qc_metrics(qc_vars=["mt"], percent_top=None, log1p=False)`。
4. 保留满足 `200 < n_genes_by_counts < 2500` 且 `pct_counts_mt < 5` 的细胞，质控后为 **2638 × 13714**。
5. 在归一化前执行 `adata.layers["counts"] = adata.X.copy()`。

### 2. 归一化与高变基因

1. 执行 `sc.pp.normalize_total(adata, target_sum=1e4)`。
2. 执行 `sc.pp.log1p(adata)`，并保存 `lognorm` 层供 marker 分析使用。
3. 使用原始 `counts` 层选择 **2000** 个高变基因，固定参数为：
   - `flavor="seurat_v3"`；
   - `n_top_genes=2000`；
   - `min_mean=0.0125`；
   - `max_mean=3`；
   - `min_disp=0.5`。
4. 固定细胞条形码、高变基因标识及顺序。

### 3. 模型输入 `X_scaled`

只在选定的 2000 个高变基因上构造模型矩阵：

- 回归 `total_counts` 和 `pct_counts_mt`；
- 执行 `scale(max_value=10)`；
- 得到 **2638 × 2000** 的有限矩阵，取值范围为 `[-10,10]`；
- `X_scaled` 是 PCA 和 CSP 的统一模型输入；
- CSP 使用 `center=False`，不再额外居中；
- `lognorm` 仅用于 Wilcoxon marker 分析；
- 稠密数组仅针对 2000 个高变基因生成。

固定预处理输出：

- `PBMC/data/notebook_scaled_preprocessed.h5ad`
- `PBMC/data/notebook_scaled_preprocessed_model_inputs.npz`
- `PBMC/figures/notebook_scaled_qc/`

## 三、PCA 参照

实现文件：`PBMC/PCA.py`。固定配置为：

- `n_comps=50`；
- `zero_center=True`；
- `svd_solver="arpack"`；
- `random_state=0`；
- 前 40 个主成分构建欧氏邻域图；
- `n_neighbors=10`；
- Leiden `resolution=0.7`、`flavor="igraph"`、`n_iterations=2`、`directed=False`。

保存方差比例、载荷、表示、邻域图、UMAP、Leiden 标签和 top-5 markers。当前 PCA 参照结果为 **8 簇**。

固定输出：

- `PBMC/data/notebook_scaled_pca_baseline.h5ad`
- `PBMC/data/notebook_scaled_pca_baseline_scaled_top5_markers.csv/.md`
- `PBMC/figures/notebook_scaled_pca/`

## 四、CSP 拟合与 RSP

实现文件：`PBMC/CSP_GS.py`。固定配置为：

| 参数 | 固定值 |
|---|---:|
| `H` | 50 |
| `alpha` | 5 |
| `a_theta`, `b_theta` | 2, 2 |
| `theta_inf` | 0.05 |
| `a_sigma`, `b_sigma` | 1, 1 |
| `center` | False |
| `n_iter` | 20000 |
| `burn_in` | 10000 |
| `thin` | 10 |
| `seed` | 2026 |
| 保留样本 | 1000，对应 sweep 10010–20000 |
| 条件均值 `batch_size` | 32 |

RSP 是正式流程的必需步骤，沿用包内默认配置并保存 `return_transforms=True` 的结果。保留 RSP 状态、损失历史、varimax 迭代数和警告；状态 4 直接失败，状态 3 不得标记为完整验证通过。

不移植 Algorithm 2，不引入自适应 `H`。

CSP 原生 `H*` 只在原始先验坐标系中解释，不在 RSP 后重新计算。同一保留位置的 `B` 与 `H*` 相差半个 sweep，只分别做边际汇总，不作逐样本联合解释。

## 五、全部 50 列加权 score

对每个保留参数样本计算条件均值：

```math
M_s=(I_H+B_s^\top\Omega_s^{-1}B_s)^{-1}B_s^\top\Omega_s^{-1}X_c^\top,
```

再使用该样本自己的 RSP 变换计算 `R_s^T M_s`，分批累计并取平均，不分配完整 `(S,H,n)` 分数数组。

固定得分构造：

1. 使用 RSP 后的 `B_hat_` 计算全部 50 个载荷列的 L2 范数；
2. 用这些范数对完整对齐条件均值逐列加权；
3. 同步加权分批均值，供半链稳定性诊断使用；
4. **保留全部 50 列**，得到 **2638 × 50** 表示，不再截取前 20 列。

不得使用 RSP 前的载荷均值或范数。

## 六、9 簇聚类、标签和 marker

### 1. 固定下游参数

| 项目 | 固定值 |
|---|---|
| score 列数 | 全部 50 列 |
| `n_pcs` | 50 |
| 距离 | Euclidean |
| `n_neighbors` | 10 |
| `resolution` | 0.40 |
| `random_state` | 0 |
| `flavor` | `igraph` |
| `n_iterations` | 2 |
| `directed` | False |

当前结果由全部 50 列加权 score 和 `resolution=0.40` 得到 **9 个簇**。既有分辨率搜索记录保存在 `uns['leiden9_d50']`，其中 0.35 和 0.40 均得到 9 簇，固定选择更接近原设置的 0.40。仅调整 Leiden 标签时复用相同邻域图和 UMAP 坐标。

### 2. 簇编号

- 底层 CSP/Leiden 标签和 marker 文件中的原始簇编号保持为 **0–8**；
- 聚类图的展示编号固定为 **1–9**；
- 映射为“展示编号 = 原始编号 + 1”；
- H5AD 中保存展示标签列及 `display_cluster_offset=1`，不修改原始标签或 marker 文件。

### 3. Marker 与图形

- 在全部质控基因的 `lognorm` 上执行 Wilcoxon 排序，`mask_var=None`、`use_raw=False`，并进行 BH 校正；
- 每簇输出 top 5，包括 score、log fold change、原始 p 值和校正后 p 值；
- UMAP 仅用于展示，不单独作为算法优劣证据；
- 保存 score spectrum、RSP 前后载荷列范数、ESS、`H*` trace/histogram、代表性 covariance trace 和聚类图。

## 七、当前固定结果与诊断

| 项目 | 当前结果 |
|---|---:|
| PCA 簇数 | 8 |
| CSP 簇数 | 9 |
| score 形状 | 2638 × 50 |
| `H` | 50 |
| `H*` | 49（全部保留样本） |
| RSP `status_code` | 1 |
| 与 PCA 的 ARI | 0.6388 |
| 与 PCA 的 AMI | 0.7910 |
| 最小 bulk ESS | 约 1.326 |
| `validation_status` | `review_required` |

当前结果保留两项统计警示：

1. 所有保留样本均为 `H*=H-1`，说明截断饱和，原生维度后验和截断尚未得到验证；
2. 最小 bulk ESS 很低，采样可靠性不足。

因此，固定当前工作流程不等于宣布采样验证通过，也不应将 9 簇直接解释为真实细胞类型数。不得根据聚类表现反向选择 CSP 先验或截断参数。

## 八、固定输出

```text
PBMC/data/pbmc3k_raw.h5ad
PBMC/data/notebook_scaled_preprocessed.h5ad
PBMC/data/notebook_scaled_preprocessed_model_inputs.npz
PBMC/data/notebook_scaled_pca_baseline.h5ad
PBMC/data/notebook_scaled_pca_baseline_scaled_top5_markers.csv/.md
PBMC/data/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50.h5ad
PBMC/data/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50_model.npz
PBMC/data/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50_top5_markers.csv/.md
PBMC/figures/notebook_scaled_qc/
PBMC/figures/notebook_scaled_pca/
PBMC/figures/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50/
PBMC/PBMC.ipynb
```

正式 CSP 输出分为 H5AD 和独立模型 NPZ，保存原始参数样本、`H*`、RSP 变换、全部 50 列 score、载荷范数、分批均值、采样诊断、配置和软件版本。

其他 `d10`、`d20`、中间 `d50`、pilot 和 alpha/theta 敏感性实验产物不属于本固定流程。

## 九、固定复现入口

在包含 PBMC 分析依赖的 Python 环境中运行：

```bash
python PBMC/data_processing.py PBMC/data/pbmc3k_raw.h5ad \
  --output PBMC/data/notebook_scaled_preprocessed.h5ad \
  --qc-figure-dir PBMC/figures/notebook_scaled_qc

python PBMC/PCA.py PBMC/data/pbmc3k_raw.h5ad \
  --output PBMC/data/notebook_scaled_pca_baseline.h5ad \
  --figure-dir PBMC/figures/notebook_scaled_pca

python PBMC/CSP_GS.py PBMC/data/notebook_scaled_pca_baseline.h5ad \
  --output PBMC/data/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50.h5ad \
  --figure-dir PBMC/figures/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50

python -m jupyter nbconvert --to notebook --execute --inplace PBMC/PBMC.ipynb
```

固定验收要求：

- 预处理维度、HVG 数量和 `X_scaled` 范围正确；
- PCA 与 CSP 使用相同的细胞和 `X_scaled`；
- 保留 1000 个 CSP 样本，sweep 为 10010–20000；
- RSP 变换近似正交，载荷满足 `B_aligned[s] ≈ B_samples[s] @ R_s`；
- 最终 score 为全部 50 列，形状为 `2638 × 50`，全部有限且无退化列；
- Leiden 使用 `resolution=0.40`，原始标签为 0–8，展示标签为 1–9；
- marker 来自完整 `lognorm`，top-5 文件与 H5AD 标签一致；
- H5AD、NPZ、图形、诊断和已执行 notebook 均可重新读取；
- `H*` 饱和、低 ESS 和其他采样警示均据实保留。

当前工作流程、参数和输出以此文件为准。
