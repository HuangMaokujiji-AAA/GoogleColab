# Rare3D-FM 本地运行版

从本地保存的 Colab HTML 中恢复代码，保留本征 VAE、异配微环境编码器、稀有原型、RW-UOT、四叉树分块、RareFlowNet 和 Heun 重建流程。原始网页仅在本地保留作参考，不随代码上传；运行入口是 `run.py`，不依赖 Notebook、Google Drive 或 Colab。

## 运行

当前项目已创建 `.venv`，可以直接从项目目录运行：

```bash
# 只查看本地数据，不训练
.venv/bin/python run.py --inspect

# 小规模真实数据试跑：3 个切片，每片最多 256 个细胞，64 个基因，每阶段 1 轮
.venv/bin/python run.py --smoke-test --device cpu

# 正式运行：读取全部切片，采用 configs/local.json 中的训练轮数
.venv/bin/python run.py
```

默认数据根目录是 `/Users/huangmao/Downloads/Code/DeepSpatial/data`，程序自动读取其 `Zhuang-ABCA-2/` 下的 `.h5ad`。也可以直接传入切片目录：

```bash
.venv/bin/python run.py \
  --data-dir /Users/huangmao/Downloads/Code/DeepSpatial/data/Zhuang-ABCA-2 \
  --output-dir outputs/experiment_01 \
  --device cpu
```

`auto` 按 CUDA → Apple MPS → CPU 的顺序选择可用设备；也可显式传入 `--device cpu`、`cuda` 或 `mps`。本机已完成 CPU 小规模验证，CUDA/MPS 需要在对应硬件上验证。全量运行包含 250 轮 VAE、200 轮微环境训练和 400 轮流匹配训练，耗时取决于设备；试跑结果仅用于检查流程，不能作为模型质量结论。

在其他机器创建独立环境（Python 3.10–3.12）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --smoke-test
```

当前 `.venv` 复用了本机 `d2l` 环境的 PyTorch、Pandas 和 Matplotlib，并在项目内补充了 AnnData、HDF5 和可加载的 SciPy/scikit-learn；没有修改原 Conda 环境。迁移到其他机器时按上述命令重新安装。

## 文件职责

```text
run.py                       命令行入口
configs/local.json           数据路径、设备、训练与内存参数
rare3d/
  config.py                  配置读取、随机种子与设备选择
  data.py                    切片读取、基因对齐、深度识别与归一化
  models.py                  VAE、微环境编码器、强度预测器、RareFlowNet
  transport.py               RW-UOT、四叉树、稀有锚点先验与 OT 配对
  training.py                VAE → 原型发现 → 微环境 → 流匹配训练
  reconstruction.py          分块 Heun 积分、表达解码与增量 H5AD 导出
  evaluation.py              稀有恢复指标与对比图
  pipeline.py                各阶段的顺序调度与结果保存
tests/test_local_pipeline.py 本地迁移关键回归检查
```

## 配置

直接修改 `configs/local.json`，或创建仅含覆盖项的 JSON，再传 `--config`。例如先做一个较小的实验：

```json
{
  "output_dir": "outputs/pilot",
  "max_sections": 5,
  "max_cells_per_section": 5000,
  "vae_epochs": 10,
  "niche_epochs": 10,
  "fm_epochs": 20,
  "max_vae_steps_per_epoch": 20,
  "max_niche_steps_per_epoch": 20,
  "max_fm_steps_per_epoch": 20,
  "max_cells_per_tile": 256,
  "max_context_cells": 512,
  "niche_context_cells": 1024
}
```

```bash
.venv/bin/python run.py --config configs/pilot.json
```

需要自行将上面的 JSON 保存为 `configs/pilot.json`。`max_sections` 从文件序列中等间隔选择切片；`max_cells_per_section` 随机抽取细胞。两项默认 `null`，表示全部读取。减小 `batch_size`、`max_cells_per_tile`、`max_context_cells` 和 `niche_context_cells` 可降低单批内存。全量原始表达矩阵仍以 CSR 保留在主机内存中，因此总内存随数据量增长。

`expression_mode` 默认 `auto`，抽查各切片的非零表达值是否为整数来选择 ZINB 或 Hurdle-Gaussian；若整数值经过预处理，可手动指定 `counts` 或 `normalized`。

`--skip-export` 只训练和验证，不生成完整三维表达体积。`--smoke-test` 会覆盖模型维度、轮数和数据规模，并默认写入 `outputs/smoke/`；通过 `--output-dir` 可改到独立目录。

## 输出

正式结果默认写入 `outputs/local/`：

- `config.json`：此次实际使用的参数。
- `data_summary.json`：切片、细胞、基因、物理深度和验证方案。
- `models.pt`：VAE、微环境、强度预测器、流网络权重，以及基因、原型、空间尺度等元数据；在最终重建前保存。
- `metrics.json`：验证指标；两切片输入会记录跳过验证，空重建会记录原因。
- `rare3d_val_comparison.png`：有非空验证重建时生成。
- `zhuang_abca2_rare3d_reconstructed.h5ad`：所有观测切片和相邻切片之间的中间重建切片。

H5AD 的 `X` 是 CSR 表达矩阵；`obsm['spatial']` 保存平面坐标，`obsm['spatial_3d']` 保存还原顺序的三维坐标。`obs` 中有来源、切片、深度和观测细胞类型；生成细胞的类型记为 `unknown`。观测值与解码后的期望表达值通过 `source` 区分。导出时逐片追加表达矩阵，完成后才将 `.partial.h5ad` 更名为正式文件。

`models.pt` 保存最终权重和元数据；目前入口每次执行都会重新训练，不提供中断后继续训练。重新使用同一个输出目录会覆盖该次结果，建议每个正式实验设置独立的 `output_dir`。

## 与原 Colab 的适配差异

- 直接读取已解压的本地文件，移除安装单元、Drive 挂载、ZIP 搜索和 `/content` 路径。
- 优先识别 `x_ccf/y_ccf/z_ccf` 或 `x/y/z`。本地这批数据使用 `obs['z']` 的真实深度，避免把文件序号当作深度。只有二维坐标时才从切片文件名提取序号作为深度。
- H5AD 表达矩阵按批读取后转为 CSR；空间近邻改用精确 KD-tree，避免全切片平方距离矩阵。微环境训练对参考细胞和查询细胞设上限；微环境推理分批计算，但使用完整切片的邻居。
- 流训练和重建的参考上下文设上限；单侧有细胞的区域从实际观测锚点生成先验。四叉树采用实际坐标范围，并保证验证切片不会进入微环境或流匹配训练三元组。
- 修复原 Heun 求解器最后一步退化为 Euler 的问题；每一步均使用预测与校正。
- 评估只将 Marker 子矩阵转为稠密矩阵，导出逐片追加；验证重建为空时保存明确状态。

验证口径延续原项目的**传导式评估**：中间切片不参与微环境和流网络训练，但参与 VAE、基因筛选、尺度和原型统计。因此当前指标不代表完全未见切片的独立泛化表现。生物学标签不作为训练目标。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py --smoke-test --device cpu
```
