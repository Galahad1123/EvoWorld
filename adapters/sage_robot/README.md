# SAGE 专家轨迹生成

[English / 原始实验详记](README.en.md) · [项目首页](../../README.md)

## 模块职责

保留原版 M2T2 抓取候选生成、cuRobo 碰撞规划、SAGE IsaacLab 状态机和 HDF5 采集器。

## 使用与配置

优先使用仓库统一入口以及机器自己的 `configs/local.yaml`。资源、解释器和上游源码目录由配置指定；权重、真实照片、实验输出与密钥不进入 Git。

当前安装和下载方法分别见[环境说明](../../docs/ENVIRONMENTS.md)与[资源清单](../../docs/RESOURCES.md)。历史详记中的绝对路径、主机名和 phase 目录仅用于理解原始实验，不应直接复制为新机器配置。

## 验证重点

PointNet++ CUDA、真实 M2T2 输出、cuRobo 轨迹，以及当前场景中的完整专家示范。

模块的历史测试数值与逐次实验记录保留在英文原始详记中；最新部署验收以[验收说明](../../docs/ACCEPTANCE.md)和对应运行报告为准。诊断、dry-run、真实仿真成功和训练效果必须分别报告。
