# 流水线概览图：布局与依据

## 布局

横向五列，依次为：Photo-to-Scene、Simulation、Expert Demonstrations、Dataset Bridge、Policy Learning。
每列采用浅色圆角容器，上方标题、中间 imagegen 图标、下方两行说明。文字与图标独立布局，不将整张架构图交给图像模型生成。

连接为单向柔和箭头：照片与任务 → GPT 辅助建模/资产迭代 → USD/Isaac 场景 → SAGE 专家采集 → HDF5/LeRobot 与动作转换 → RLinf SFT → DSRL → Evaluation。
为保持简洁，将 GPT 建模归入第一列，场景归入第二列，SFT/DSRL/评测归入第五列。图中不表示这些环节具有相同完成度，也不把评测回接到 GPT 形成不存在的自动优化闭环。
下方统一说明：camera / action / normalization contracts。

## 静态代码依据

- evoworld/cli.py：generate、collect、convert、sft、rl、eval 的统一调度。
- generate → adapters/realtosim_photo/agent.py。
- collect → adapters/milk_bin/collect.py，调用 SAGE 原始采集器并适配场景。
- convert → adapters/milk_bin/export_dataset.py。
- sft → RLinf/examples/sft/train_vla_sft.py；正式 SFT 与 RL 均由 RLinf 执行，不把 OpenPI 另画为必经训练器。
- eval → adapters/milk_bin/evaluate.py。
- configs/local.example.yaml 与 docs/SIDE70_PIPELINE.md：侧移 7 cm 观测版本、独立环境与资源约定。

现有 side70 文档记录 SFT 4/20 与 DSRL 9/20；展示前继续核对远端原始报告。图标只是概念插图，真实场景图片另列。
