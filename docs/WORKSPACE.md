# 工作区布局

## P6000 正式入口

2026-11-10 起，直接打开 **`/root/autodl-tmp/EvoWorld/`**。该目录本身就是 EvoWorld Git 仓库，不再嵌套 `code/`。

```text
EvoWorld/
├── evoworld/              # 统一入口
├── adapters/              # 已纳入版本管理的任务与组件适配
├── configs/               # 版本化配置与忽略的机器配置
├── third_party/RLinf/     # 唯一 RLinf 开发入口，Git submodule
├── research/rl/           # 算法实验配置、评测、报告
│   └── outputs/           # 正式 RL 研究结果，不入 Git
└── storage/               # 本机资源，不入 Git
    ├── envs/              # 已有虚拟环境
    ├── models/            # 共享权重
    ├── assets/            # 仿真资产
    ├── cache/             # 下载缓存
    ├── upstream/          # 使用中的专家和 Isaac 依赖源码
    ├── runtime/           # 专家运行所需系统库
    ├── runs/              # 通用流水线的新运行输出
    ├── archive/           # 旧代码、配置、历史实验及诊断脚本
    └── compat/            # 仅用于维持历史绝对路径
```

原来的外层 adapters、configs、runtime/libero、旧 RLinf 源码和历史 side70 实验均在 `storage/archive/pre-restructure-20261110/`。专家采集实际需要的 `sage_robot/src` 单独保留为 `storage/upstream/sage-expert/`，不归入废弃代码。历史检查点和数据没有删除。

## 旧路径约定

`/root/autodl-tmp/evoworld-dsrl` 现在仅是指向 `storage/compat/p6000-oldroot` 的兼容链接。既有 `/home/hello/lizk/vlas/EvoWorld` 别名也经此入口继续解析。请使用新正式目录开发，不在兼容目录修改代码。

保留兼容入口是因为虚拟环境的 shebang、editable 安装、资产链接和历史记录仍含旧绝对路径。不要直接删除 compat，也不要批量替换历史 provenance。环境解释器必须保持 `storage/envs/<env>/bin/python`，不能解析成其基础 Python 目标。

## 结果位置

LIBERO 基线：`research/rl/outputs/runs/libero/pi05-sft/baseline-pi05-20k-20261110/`。原结果路径仍可访问。迁移按目录重命名完成，不复制大权重；78 个结果文件在迁移前后逐一核对哈希。

配置仍使用 `configs/local.yaml`；本机资源和迁移记录不提交 Git。目录迁移记录位于 `storage/workspace-migration-20261110.json`。迁移仅在 P6000 执行；5090-1 保持原部署位置。
