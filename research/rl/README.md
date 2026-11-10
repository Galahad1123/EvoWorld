# RL 算法研究

EvoWorld 负责实验与评测，仓库根的 `third_party/RLinf`中的 fork 负责算法实现。当前只完成架构迁移和 LIBERO 独立基线归档，尚未开始新的算法复现或原生 RL 训练验收。

## 源码与版本

```bash
git clone --recurse-submodules https://github.com/Galahad1123/EvoWorld.git
# 已有克隆：确认子模块无未保存修改，再执行
git submodule update --init --recursive
```

RLinf fork 为 `https://github.com/Galahad1123/RLinf`。`evoworld/compat` 保留现有兼容修改，新算法从明确基点新建开发分支。submodule 默认 detached HEAD；开发前必须创建分支。禁止在正式实验运行期间切换源码。

先提交并推送 RLinf 的修改，再更新 EvoWorld 的 gitlink 和 `manifests/upstreams.json` 中的 commit，最后推送 EvoWorld。不能只推父仓库引用却遗漏子仓库提交。原 RLinf patch 已并入 fork，不再重复应用；其他第三方组件暂维持原有管理方式。

## 目录边界

- `configs/algorithms`、`configs/benchmarks`、`configs/experiments`：算法配置、任务协议与具体实验组合，目前预留，不代表算法已复现。
- `scripts`：轻量启动及独立评测，不另造训练框架。算法主体、worker、runner、采样和 loss 修改放入 RLinf fork。
- `protocols`：公平评测和产物保留规则。
- `reports`：可入 Git 的精简结果分析。
- `outputs/runs/<benchmark>/<algorithm>/<run_id>`：正式配置、指标、日志、视频和检查点；不入 Git。
- `outputs/scratch`：有归属标记的临时测试产物，验收成功后清理；失败证据保留。

共享模型、环境和资产不迁入仓库，不复制大权重。`configs/local.yaml` 的 `research.output_root` 可指定磁盘位置；未设置时默认本模块 `outputs`。相对路径仍基于配置文件目录。

## LIBERO 独立评测入口

机器配置增加 `environments.libero`（现有 LIBERO 环境解释器）、`resources.libero_checkpoint`（包含 `model_state_dict` 的 20k 目录），并令 `upstreams.RLinf` 指向 submodule。现有环境可以复用，LIBERO overlay 仍依赖原 `rlinf-py312` 环境，不可删除后者。

```bash
python research/rl/scripts/eval_libero.py --config configs/local.yaml --run-id my-baseline --stage env --dry-run
```

确认路径后依次执行 `env`、`smoke`、`full`，前两阶段 `--num-envs 2`，正式阶段 `--num-envs 25`。本轮迁移不执行这些仿真命令。正式结果不可覆盖；新的运行必须使用新 ID。已有 500 回合基线报告见 `reports/libero-pi05-20k-20261110.md`。

正式 RL 训练后续使用 RLinf 原生入口；此独立推理脚本不代表完整训练链路已验收。

## RECAP Task 0

[实验配方、资源来源和运行状态约定](RECAP.md)。正式运行是否完成以对应 outputs 目录的 pipeline-status.json 为准。
