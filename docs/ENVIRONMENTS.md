# 环境安装与源码准备

[English](ENVIRONMENTS.en.md) · [返回首页](../README.md)

## 原则

使用独立环境承载 GPT、专家生成、RLinf 和格式转换。共享可兼容的 Isaac 运行库，不复制整个虚拟环境，不就地升级已验证的 RLinf 环境。

轻量入口只要求 Python ≥3.10 与 PyYAML。机器配置填写真实虚拟环境解释器路径；不要将解释器符号链接解析成虚拟环境之外的基础 Python。

## 固定上游版本

先查看准备计划，再显式执行：

```bash
python scripts/prepare_sources.py --destination /your/storage/upstream
python scripts/prepare_sources.py --destination /your/storage/upstream --execute
```

工具按 `manifests/upstreams.json` 克隆固定 commit 并应用补丁。已有 checkout 版本不匹配时停止，不自动 reset。`sage-expert` 是独立的 SAGE IsaacLab、M2T2、server 与 cuRobo 兼容源码副本。

## 各环境职责

| 环境 | 用途与要求 |
|---|---|
| GPT 请求环境 | OpenAI SDK、PyYAML；5090 当前使用 SAGE 环境。最小 g1-realtosim 环境仅包含 USD／几何工具，不能据此认为已经装好 GPT SDK。 |
| 专家生成环境 | Torch、M2T2、PointNet++、cuRobo、nvdiffrast 及 SAGE 的 IsaacLab 命名空间。CUDA 扩展须在目标软件栈中编译和验证。 |
| RLinf/SFT/转换 | 固定版本的 RLinf、OpenPI、LeRobot；P6000 已有 Python 3.12、Torch 2.11 CUDA 13.0 环境。迁移不要求再单独训练一次 SFT。 |
| Isaac 运行库 | 可以独立解释器安装，也可以由其他环境加载。通过 `environments.sim` 和可选 `isaac_experience` 指定，不在代码中固定 Python 小版本目录。 |
| RoboTwin 等基准 | 可选独立后端，需要其兼容分支、资产和模型协议；不会随当前任务自动安装。 |

专家环境版本清单见 `configs/expert-packages.txt`。它记录已验证源环境自身的普通 Python 包，不包括共享的 Isaac 库和需要源码编译的扩展。必须先准备兼容的 Torch/CUDA，再安装这些包和扩展；不能将该文件当成覆盖所有组件的通用 lock。

历史 `adapters/rlinf/requirements-frozen.txt` 是环境证据，不是可直接跨机安装的依赖锁。其 editable 安装位置须改为目标 checkout。

## 两机分工与凭据

GPT 建模固定在 5090-1；P6000 只消费生成并验证的场景包。P6000 无需复制 GPT 密钥、代理配置或建模环境。

只有 GPT 请求进程加载凭据。`realtosim_photo/build_candidate.py` 执行生成程序前清理凭据变量。下载可使用 hf-mirror 和清华源；镜像不会改变资源版本和校验值。

## 验证范围

`doctor` 检查路径及模块可发现性，不启动 GPU 推理。专家环境需要额外验证 PointNet++ CUDA、M2T2 输出、cuRobo 规划和真实示范采集；不能只凭 `import` 成功验收。

更新 CUDA、Torch 或 Isaac 时保留原环境，并重新执行相关兼容性检查。NVIDIA 软件与资产遵循各自许可；统一入口不代替用户接受许可。

## RLinf 算法研究架构

RLinf 已改为 fork + Git submodule 管理，源码位于 `third_party/RLinf`，不再应用 RLinf 补丁。算法研究独立组织在 `research/rl/`；环境、权重和数据继续通过机器配置引用。详见 [研究模块说明](../research/rl/README.md)。

RLinf 安装兼容入口 / installer: `bash third_party/RLinf/requirements/install-evoworld.sh --help`。原始 `install.sh` 保留上游内容；旧 `rlinf-installer.patch` 已移除，请勿继续引用。已有可用环境无需重新安装。
