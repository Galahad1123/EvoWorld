# 燕麦奶入盒：π₀.₅ SFT 基线

执行主机为 `5090-1`，项目根目录为 `/path/to/EvoWorld`。本次实验保存在 `experiments/phase8-milk-bin-20261108`。

使用 GPT 根据新照片生成的蓝色空心收纳盒、已有燕麦奶资产、office05 背景和用户确认的 v4 tabletop / wrist 相机。专家仍使用 M2T2 与 cuRobo，整体保持 0.8× 速度；运动阶段切换时停顿 0.3 秒，投放到位后固定末端目标，再松爪。

正式数据为 30 条成功训练示范和 5 条独立验证示范。只有当前 `train/` 和 `validation/` 可用于正式实验；pilot 与 rejected 目录仅用于诊断。每条示范必须通过物理验收、停顿检查、动作与图像时序检查，才可转换和训练。

π₀.₅ 使用官方通用基座。VLM 冻结，动作专家保留 FP32 主参数并参与训练。输入为两路 224×224 RGB，以及实测的 7 个关节角和夹爪宽度。动作是机器人基坐标系下的平移增量（米）、空间旋转向量（弧度）及夹爪指令（+1 张开，−1 闭合）。物体的真实状态只用于奖励和终止判定。

当前 SFT 设置：global batch = micro batch = 4；20,000 个优化器更新步；预热 1,000 步；每 3,000 步及最终第 20,000 步保存。短测平均约 0.51 秒/更新步，训练进程峰值显存约 16.7 GiB，预计正式 SFT 约需 3 小时，实际以日志为准。

`collection.pid`、`collection-current-stage.txt`、`pipeline.pid`、`pipeline-stage.txt` 和退出状态文件用于跟踪后台任务。`after_collection.sh` 会等待采集完成，重新验收全部示范，导出数据和专家视频，然后训练、检查冻结参数、运行 20 个独立种子的在线基线评测。任何验收失败都会阻止后续训练。

成功要求纸盒位于盒内、底部得到支撑、速度足够低，并连续 1 秒保持张开指令且两指位置均大于 39 mm。允许纸盒高出盒沿。盒子外尺寸 90×95×55 mm 来自照片估计，并非实测值。

任务通过 RLinf 扩展注册为 `EvoWorld-Milk-Bin-Franka-v0`，在独立 RLinf 环境中复用 SAGE 的 Isaac Lab 控制器，目前验证了单环境。证据文件包括 `native-sft-batch4-report.json`、`native-checkpoint-verification-final.json`、`native-wrapper-smoke/report.json` 和 `online-smoke/report.json`。最后一个只是 10 步推理链路检查，不能作为训练后成功率。

校准评测使用种子 20000–20019，正式配对评测保留 30000–30049。5 条监督验证示范独立保留。DSRL 最小链路已于 2026-11-08 在 D46-6000x1 验证通过；正式训练仍等待 SFT 基线能力确认，4 小时预算尚未开始。实施记录见 `WORK_LOG.md`。

## DSRL 最小链路验收（2026-11-08）

使用 D46-6000x1 的 RTX PRO 6000 Blackwell 96GB，独立克隆现有 RLinf Python3.12 / Torch2.11cu130 与必要 Sim6 运行时；未改系统环境，5090-1 的正式 SFT 保持运行。仅使用诊断用 SFT 第3步检查点，不能据此评价任务成功率。

执行 `bash adapters/milk_bin/run_dsrl_smoke.sh`：原生 RLinf EmbodiedRunner + DSRL/SAC，单个真实燕麦奶入盒环境，两路224RGB，batch4/4。3轮各40控制步，共24个执行动作块、3条短轨迹；终止观测另做3次推理，共27次策略调用。先完成4次SAC更新，保存检查点，再由独立进程恢复并继续2次更新，总计6次。全部损失/梯度有限；3,353,433,872个基座参数哈希不变，4,857,226个可训练头参数所属的actor/critic及熵系数发生更新。

恢复核验覆盖模型、target、actor/critic/alpha优化器、alpha、完整FP32 EMA、回放内容和RNG、训练端Torch/CUDA RNG、调度器及更新计数，逐项一致。并未恢复仿真物理状态或rollout进程RNG，不声称未来轨迹逐位一致。原生入口和SAC算法保留；`dsrl_entry.py`在Ray复制worker方法前注册扩展，`dsrl_checkpoint.py`补存原生遗漏的计数/EMA/RNG，`dsrl_precision.py`保持alpha为FP32以避免BF16吞掉小更新。

每秒采样的总显存峰值48,851MiB（47.706GiB），当前配置无法放入单张32GB5090；未验证卸载优化。训练/恢复均正常退出0，GPU归零。7个DSRL定向测试通过。RL相机只请求策略所需RGB，SAGE专家采集的深度相机不变。

D46完整证据：`experiments/phase8-milk-bin-20261108/dsrl-smoke/{report.json,train.log,resume.log,gpu-memory.csv,audit/}`；完整检查点位于其 `native-dsrl-smoke/checkpoints/`。小型证据镜像在本机 `remote/D46-6000x1/experiments/phase8-milk-bin-20261108/dsrl-smoke/`，5090-1副本为 `experiments/phase8-milk-bin-20261108/dsrl-smoke-D46-evidence/`。首次未生效的worker核验运行单独保留为D46的 `dsrl-smoke-late-hooks/`，不作为验收结果。

限制：仅验证单环境短链路，未做任务成功率、长时稳定性、多环境吞吐或正式4小时RL。正式实验须先完成20000步SFT和20个独立种子的在线校准。

## SFT 完成后自动启动正式 DSRL

用户已授权自动交接，控制脚本为 `dsrl_handoff.py`。后台监控位于本机 `maintenance/dsrl-handoff-20261108/`，查看 `status.json`、`supervisor.log`、`supervisor.pid`；成功启动后生成 `launched.json`。使用目录权限700的SSH控制连接，不复制私钥，不在本地落盘大模型权重。

启动条件：原5090-1流水线退出0；最新完整SFT检查点为20000步且冻结参数核验通过；20个种子20000–20019的650步在线基线完整，使用同一检查点，成功率处于此前约定30%–70%。失败则记录原因并停止，不降级选用旧检查点，不自动修改任务。等待上限6小时，每个文件另有30分钟传输超时；因此总交接时间可能超过6小时。

权重和norm stats流式传输至D46 `models/pi05-milk-bin-sft-20000/`，逐文件大小/SHA256验证后原子发布。正式结果在D46 `experiments/phase8-milk-bin-20261108/dsrl-formal-20261108/`；查看 `source-manifest.json`、`started.json`、`stage.txt`、`train.log`、`audit/`、`exit-status.txt`。配置 `dsrl_formal.yaml`：原生DSRL/SAC，1环境、650步、双224RGB、batch4/4，每轮32次SAC更新，回放保留32条轨迹；32×4约等于每轮新采样130个动作块的数据量。学习率和噪声网络与已通过的短测一致。

训练墙钟预算4小时，约3小时55分后在完整更新轮次边界保存退出，另有4小时硬超时。每10轮保存，仅保留最近两份完整RL检查点，SFT和历史短测结果不受影响。`dsrl_budget.py`只包装原生runner的完成轮次/保存钩子；`dsrl_cleanup.py`按本次独有环境标记清理残余进程，避免全局ray stop。若硬超时打断保存，可能留下未完成目录；以 `completed.json` 为完整检查点标识。

本次自动衔接不修改任务或扩大预算，不使用30000–30049正式留出种子。训练收益仍需后续配对在线评测；训练成功启动不等于任务改善。13个定向测试通过；SSH真实传输/错误哈希拒绝、Hydra正式配置解析另作实际检查。独立审查的进程清理问题已修复；尚未为全部交接场景建立独立自动化集成测试。
