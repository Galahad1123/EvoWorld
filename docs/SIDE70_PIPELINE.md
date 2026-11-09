# 侧移7厘米：P6000完整训练流程

本轮状态：已启动新视角专家采集；SFT、基线、DSRL尚未完成。实际进度以主机状态文件为准。

用户确认旧中央视角错误，因此正式任务现在固定侧移7厘米。位置 `[0.07, 0, 0.035]` 为手部坐标系米制偏移，朝向沿用已确认预览。tabletop不变。机器、场景、数据来源和检查点均使用 `milk-bin-side70-v1`，旧中央数据或权重不会被重新标记后混入。

## 本轮预算

- 专家：30条训练示范、5条独立验证示范，种子1610/1710；最多150/30次尝试。0.8倍速，阶段停顿至少0.3秒。采集完成后独立检查物理、时序和停顿，再转LeRobot。
- SFT：官方π₀.₅基座，VLM冻结，global/micro batch均为4；20000更新步、1000预热步、每3000步及最终保存。保留所有策略权重，完成后仅保留最近两份原生优化器恢复状态，避免挤占RL检查点空间。
- 在线校准：最终20000步权重，20个种子20000–20019，每次最多650控制步，前5条保存视频。
- DSRL：仅在上述验收成功且校准成功率30%–70%时自动进入，原生RLinf DSRL/SAC，单环境、batch4、最长4小时。保留最近两份完整RL检查点。未达到门槛会明确停止，不自动改任务或绕过判定。

## P6000产物

项目目录下 `experiments/side70-milk-bin-20261109/`：

| 文件或目录 | 用途 |
|---|---|
| `status.json`、`pipeline.log`、`pipeline-exit-status.txt` | 当前阶段、流水线日志、最终退出状态 |
| `collect-train.log`、`collect-validation.log` | 专家采集日志 |
| `train/`、`validation/` | 原始HDF5及物理验收证据 |
| `train-expert.mp4`、`validation-expert.mp4` | 两路相机专家示范 |
| `lerobot-train/`、`lerobot-validation/` | 新视角数据集，归一化仅使用训练集 |
| `sft.log`、`sft-dispatch/sft/side70-pi05/` | SFT日志、TensorBoard、检查点 |
| `baseline-calibration/`、`baseline-manifest.json` | 在线成功率、视频、DSRL门槛证据 |
| `rl.log`、`rl-results/` | DSRL训练日志、更新审计和检查点 |

实验控制脚本与小型产物直接通过SSH传递。GPT建模继续在5090-1进行，本轮不调用GPT。单条采集或训练启动不代表完整流程完成。
