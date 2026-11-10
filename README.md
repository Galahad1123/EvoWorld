# EvoWorld

[English](README.en.md) · **中文为默认维护语言**

EvoWorld 将 GPT 照片建模、SAGE 专家示范生成、Isaac 仿真和 RLinf 的 SFT／强化学习连接起来。当前任务是 Franka 单臂将燕麦奶放入收纳盒。

## 先阅读这些文档

- [工作区布局与 P6000 正式入口](docs/WORKSPACE.md)

- [环境安装与源码准备](docs/ENVIRONMENTS.md)
- [额外资源、下载地址与生成方式](docs/RESOURCES.md)
- [迁移验收结果与适用边界](docs/ACCEPTANCE.md)
- [P6000 专家环境配置与实时验收进度](docs/P6000_EXPERT.md)

代码仓库不包含模型权重、示范数据、真实照片、生成场景、虚拟环境或密钥。第三方源代码保留原有许可证，精确版本及补丁见 `manifests/upstreams.json`。

## 两台机器的分工

- **5090-1**：调用 GPT 完成照片建模、资产生成与修正；保留现有实验环境。
- **P6000（D46-6000x1）**：使用已生成的场景进行专家轨迹采集、数据转换、SFT 和 RL。该机器不承担 GPT API 调用。

环境就绪、资源就绪与任务验收是三个不同状态。`doctor` 只报告前置依赖，不代表已经通过真实仿真或训练验收。

## 配置与运行

1. 安装独立的组件环境；不要将 SAGE、Isaac、RLinf 的全部依赖混装到一个环境。
2. 为轻量入口安装 Python 依赖：`pip install -e .`，要求 Python ≥3.10。
3. 将 `configs/local.example.yaml` 复制为 `configs/local.yaml`，填写解释器、源码和资源位置。相对路径统一相对该 YAML 文件所在目录解析。
4. 仅在需要 GPT 的机器上配置凭据：默认 `~/.config/evoworld/credentials.env`，内容为 `OPENAI_API_KEY=...`。建议权限为 `600`。也可用 `credentials_file` 指向现有 `.env`；不要提交真实凭据。
5. 执行前置检查和 dry-run：

```bash
python -m evoworld doctor
python -m evoworld sft --dry-run
python -m evoworld --config /path/to/local.yaml sft --dry-run
```

入口包括 `generate`、`collect`、`convert`、`sft`、`rl`、`eval`。`--` 后的参数传给原始工具；`--output` 在分隔符前指定本次运行目录。dry-run 不启动下载、GPT 请求、仿真或训练。

GPT 模型及推理档位位于 `configs/gpt.yaml`。机器配置负责代理和下载镜像。API 密钥只加载到 GPT 请求进程，不传入专家采集和训练进程。

## 资源和实验版本

任务通过逻辑资源名引用本地模型、资产与数据。更换相机、动作定义或归一化统计后，应更新对应实验契约，不要让旧权重静默使用新观测。

用户已确认侧移7厘米为正式腕部相机。新实验固定使用 `milk-bin-side70-v1`，重新采集和训练；旧中央视角结果只作历史记录。

历史 adapter 文档保留原始实验记录；固定主机的自动启动脚本归档为文本，位于 `docs/legacy-launchers`。当前部署优先使用统一入口。代码更新不覆盖外部环境和实验结果。

本轮参数与运行入口见[侧移7厘米完整实验](docs/SIDE70_PIPELINE.md)。

## RLinf 算法研究架构

RLinf 已改为 fork + Git submodule 管理，源码位于 `third_party/RLinf`，不再应用 RLinf 补丁。算法研究独立组织在 `research/rl/`；环境、权重和数据继续通过机器配置引用。详见 [研究模块说明](research/rl/README.md)。
