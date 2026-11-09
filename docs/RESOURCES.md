# 额外资源、下载地址与生成方式

[English](RESOURCES.en.md) · [返回首页](../README.md)

Git 仓库只管理代码与配置。权重、照片、场景、示范数据和环境单独准备，并在忽略提交的 `configs/local.yaml` 中设置 `resources` 映射。`doctor` 不会自动下载；下载中的 `.partial` 文件必须校验通过后才启用。

## 官方可下载资源

| 资源 | 下载地址 | 版本及注意事项 |
|---|---|---|
| π₀.₅ 基础权重 | `gs://openpi-assets/checkpoints/pi05_base`；[OpenPI 官方说明](https://github.com/Physical-Intelligence/openpi) | 这是 GCS 目录，不是可浏览的 HTML 页面。下载完整 JAX 权重后，用固定版本 RLinf 的 `rlinf/utils/ckpt_convertor/openpi/jax_to_openpi.py` 转换；它不是现成 PyTorch 权重的下载链接。配置为 `base_model`。 |
| M2T2 | [官方模型页](https://huggingface.co/wentao-yuan/m2t2)；[固定版本文件](https://huggingface.co/wentao-yuan/m2t2/resolve/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth) | revision：`a20325dcd19cd1b838274974d0ebe35bfe383796`。可将域名改为 `hf-mirror.com`，其余路径与版本保持不变。配置为 `m2t2`。 |
| SAGE-10k | [固定版本数据集](https://huggingface.co/datasets/nvidia/SAGE-10k/tree/d425de49ff445fd3903d790a11461d23ffd0c7dd) | 当前背景来自 `layout_acb3ece0`。按场景选择性下载其依赖，不必下载全部一万个场景。 |
| Franka／Isaac 资产 | [NVIDIA Isaac 4.2 资产根目录](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/) | 从 `Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd` 开始准备引用资源。已有辅助下载脚本会保存逐文件 URL 和哈希，但历史脚本中的机器设置需核对后使用。配置为 `isaac_assets`。 |
| 专家源码 | [固定版 SAGE](https://github.com/NVlabs/sage/tree/fe3966013a495dc1642710de93a83ddf66e534a0)、[cuRobo](https://github.com/NVlabs/curobo/tree/d64c4b005459db10c5dd867d8b30a87d5bda9bdb) | 将 IsaacLab、M2T2、server 与 cuRobo 组成独立源码目录，应用 `patches/expert-current.patch`，并在目标环境重编译扩展。配置为 `expert_source`。 |
| nvdiffrast | [固定版本源码](https://github.com/NVlabs/nvdiffrast/tree/253ac4fcea7de5f396371124af597e6cc957bfae) | 专家场景渲染使用的可微渲染依赖；固定版本为 0.4.0。 |

M2T2 文件 SHA256：

```text
e35c3cb11e06f46c5d406bdfc756bc06f48b256dd6d638408d9a5ff13deb97fb
```

当前转换后的 π₀.₅ 模型有 667 个张量、3,353,433,872 个参数，其 `model.safetensors` SHA256 为：

```text
f6391204c480d6c54d31d63601e7963ef23301157df052a4b8183e7b019ce2f1
```

原转换脚本 SHA256 为 `e95be73afc0acf1b82a8ad0ba52a1967e083947829a2a10a50ba95c21db558d2`。更换转换脚本版本可能改变输出文件；JAX 原文件与 PyTorch 输出文件不能共用哈希。任务 `norm_stats` 必须由训练示范单独计算。

显式下载示例：

```bash
# 需要 Google gsutil；不能把该目录当成单个 HTTP 文件下载。
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base /your/model/storage/

# 可用 HF_ENDPOINT=https://hf-mirror.com 加速。
hf download wentao-yuan/m2t2 m2t2.pth \
  --revision a20325dcd19cd1b838274974d0ebe35bfe383796 \
  --local-dir /your/model/storage/m2t2
```

GPT 建模路线不需要下载 Qwen。镜像失败时不能静默切换模型版本。受限资源遵循原提供方的授权方式，凭据只保存在本机。

## 项目生成资源：没有公开下载地址

| 资源 ID | 内容与取得方式 |
|---|---|
| `photo_scene` | 包含 `scene_spec.json` 及 USD、网格、纹理的照片重建场景包。复用已验收结果，或在5090执行 GPT 生成→构建→渲染→反馈修正→物理检查。真实照片不公开。 |
| `storage_box` | 根据照片和已知尺寸生成的收纳盒 USD；保留引用资产和生成记录。 |
| `office_background` | 从固定版本 SAGE-10k 派生的 office 背景，含用户选定的灰色地板。只下载原场景不会自动包含这一修改。 |
| `env_template` | 采集与仿真所需的 SAGE `env_init.yaml`，须与其场景依赖一起保存。 |
| `train_data`／验证数据 | 原版专家流水线生成 HDF5，经过成功条件与停顿检查，再由 `export_dataset.py` 转换。保持数据划分和训练集归一化统计。 |
| `checkpoint` | 本地训练得到的 SFT 权重。必须显式选择，并匹配机器人、动作和相机约定；不默认选取目录中最新文件。 |

当前私有场景包位于 `assets/evoworld/milk-bin-v1/`，包含 `photo_scene/`、`box/`、`office/`、`fixture/`、相对路径版 `env_init.yaml` 和190个文件的 SHA256 清单。它不进入公开仓库。迁移后的当前任务不再从历史 phase 输出目录读取这些场景资产。

## 可选基准

[RoboTwin RLinf_support 分支](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support)与[RLinf 官方评测说明](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html)提供其环境和资产准备方法。基准不会随核心环境自动安装；Franka 单臂任务权重不能直接当作双臂基准策略使用。
