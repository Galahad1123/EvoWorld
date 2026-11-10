# π0.5 20k + 公开数据 + Task 0 RECAP 基线

本实验使用现有 20k 权重运行 RLinf 原生离线 RECAP：回报计算 → 价值模型 → 优势标注 → CFG 策略训练 → 独立在线评测。不重建 RLinf 训练器，不将结果冒充官方 48.8% 初始化的数值复现。

## 数据与权重

- [公开数据](https://huggingface.co/datasets/RLinf/RECAP-Libero10-Task0-48succ-Data)，固定 `75b382d2c066bcedd8b030285b45c856913f1497`，86.89 GB。
- `libero10_task0_sft` 实际是 10 个任务各 3 条示范，共 30 条/8005 帧；不是纯 Task 0 示范。保持公开包混合方式，不按局部 task_index=0 错误筛选。
- rollout：4096 条/1561874 帧；价值验证：64 条/25493 帧。二者均为 Task 0。SFT 为 LeRobot v2.1，rollout/eval 为 v2.0，已有环境实测可读。按各自元数据帧率取 action chunk，不额外重采样。
- [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-224)，revision `78e403963a4f6a3640d07803284752326fdf4edf`。
- [Gemma3-270M](https://huggingface.co/google/gemma-3-270m)，revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`；需已有 HF 授权。代码只读取凭据，不输出密钥。
- [π0.5 few-shot](https://huggingface.co/THU-SIGS-EILAB/Pi05_few_shot_libero)，沿用现有 20k 单份权重和归一化统计。813 个参数键与 CFG 模型全部匹配。

下载使用 hf-mirror、单份文件、固定 revision 和逐文件哈希。公开数据位于 `storage/datasets/recap-libero-task0`。每个实验复制轻量 meta，链接 data/videos；回报与优势 sidecar 只写实验视图，不修改原始元数据。

## Recipe 与输出

配置意图见 `configs/experiments/recap-task0.yaml`。`recap_prepare.py` 生成实际配置；正式目录为 `outputs/runs/libero/recap/pi05-20k-task0-20261110`。

- 价值模型：8000 步，micro/global batch=32，学习率 5e-5，warmup 500；每1000步验证全部64回合，最终价值模型用于标注。
- CFG：10000 步，micro/global batch=4，学习率1e-5，warmup1000，cosine；不使用 LoRA、不仅训练 action expert。
- 回报：gamma=1，失败终止惩罚-300。优势：N=10，top30%；SFT 标签按原实现强制为正。
- 按上游混合采样实现使用两组 weight=1；其 balance_dataset_weights 会按帧数调整采样概率，不意味着两组各占50%。
- 每3000步及最终保存原生完整检查点。此预算约产生3份价值、4份策略检查点，预留磁盘；不自动删除正式检查点。
- 独立评测：Task0固定50初始状态，每回合520步，预测horizon10/执行5步/去噪10次，10环境并行，种子195。初始化用CFG模型的 no_guide，最终用 positive、scale1；同一封装下比较，并保存双相机视频。

初始旧结果Task0为28/50，但本轮重新用CFG封装评测；不把此前整套55%当成Task0基线。小样本冒烟成功率不用于算法收益判断。

## 环境与运行

复用 `storage/envs/rlinf-libero-py312`。价值视频解码还需要 FFmpeg；本机通过 Ubuntu 镜像下载38个deb并解包到 `storage/runtime/ffmpeg`，仅22MB下载，不安装/升级系统包。包清单在 `storage/runtime/ffmpeg/debs/packages.txt`。`recap_stage.py` 为训练进程设置相应动态库目录。

```bash
# 在已有资源下载进程完成后自动顺序执行；先要求短程 acceptance.json 通过。
python research/rl/scripts/recap_pipeline.py --root /root/autodl-tmp/EvoWorld
# 失败后：保留证据，跳过成功阶段；训练阶段使用最近完整原生检查点恢复。
python research/rl/scripts/recap_pipeline.py --root /root/autodl-tmp/EvoWorld --resume
```

`pipeline-status.json` 表示当前阶段，`logs/` 保存原生输出，`provenance/` 记录提交、命令与退出码，`resources.jsonl` 记录显存与磁盘采样。任何失败都停止后续阶段，不输出成功率报告。恢复时禁止改变训练预算；部分失败的评测目录归档后重跑，不覆盖已完成评测。

测试产物位于 `outputs/scratch/recap-20261110`。验收后删除测试检查点，保留配置、日志、参数更新记录、视频与验收结果。正式实验完成状态以 metrics/summary.json 和 pipeline-status.json 为准，创建目录或启动进程不表示完成。
