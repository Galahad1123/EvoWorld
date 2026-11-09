# P6000 专家轨迹环境：配置与验收结果

[返回首页](../README.md)

## 分工

GPT 照片建模在5090-1执行。P6000不连接 OpenAI、不保存 GPT 密钥，消费已生成的场景与资产，承担专家示范、数据转换、SFT 和 RL。

## 当前已完成

- 新建独立 `sage-robot` Python 3.12 环境，复用已有 Isaac Sim 6 运行库；没有修改 RLinf 环境。
- 使用清华源安装 Torch 2.11 / CUDA 13.0、torchvision 0.26、Warp 1.13 和固定版专家依赖。
- 从 hf-mirror 下载官方 M2T2 权重，SHA256与[资源清单](RESOURCES.md)一致。
- 在本机编译 PointNet++、M2T2、cuRobo 0.7.8 和 nvdiffrast 0.4.0。
- PointNet++ 在 RTX PRO 6000 的 `sm_120` 上通过真实 CUDA 前向／反向算子测试。
- 原版 SAGE 渲染→M2T2 探针生成297个有效抓取位姿，未替换模型或预测结果。
- cuRobo 碰撞规划探针生成31个轨迹点，末端位置误差约 `4.94e-7 m`。

## 已处理的兼容性问题

1. CUDA wheel 缺少用于链接的 `libcudart.so` 名称：仅在专家环境内建立指向 `libcudart.so.13` 的相对链接，不改变系统 CUDA。
2. cuRobo 副本没有 `.git` 元数据：构建时按已验证版本设置0.7.8，保留原始源码与算法。
3. 精简 Isaac 安装没有源机器共享的绘图／coverage依赖：按源环境版本补齐。
4. P6000 的 X11 等兼容库位于项目私有 runtime目录：通过 `runtime_library_dirs` 仅添加到当前仿真进程的动态库搜索路径。

## 端到端采集验收（2026-11-09）

修复依赖后的 `collection-02` 完整运行退出码为0；第1次抓取尝试生成1条成功示范，共398帧。上限为3次尝试、1条成功示范；没有启动SFT或RL训练。

- 保留原版M2T2抓取候选、cuRobo规划和SAGE采集流程，使用已迁移的燕麦奶、收纳盒、照片桌面资产与office背景。
- 独立离线重算任务成功条件通过：位于盒内、得到支撑、已释放、线速度稳定、角速度稳定；最后1秒共51个样本全部满足。最小水平余量约10.06毫米，最大支撑高度误差约0.00060毫米，最大线速度约0.00080米/秒。
- HDF5动作与物理轨迹逐帧一致；两路RGB均为 `(398, 224, 224, 3)`、`uint8`，`next_obs`时序一致。数值字段检查通过；深度字段沿用原版uint8编码，不表示经过米制精度标定。
- 指令停顿审计通过：6段至少0.3秒的静止区间，且释放前有完整0.3秒静止指令。该审计验证命令停顿，物理稳定性由上述任务判定单独验证。
- 导出准确对应已保存示范的双相机视频，50fps、7.96秒。

这是单条示范的链路验收，不是成功率统计，也不代表策略训练或RL适配验收。沿用 `milk-bin-center-wrist-v1` 观测契约；该次验收时侧移7厘米仍是预览候选。随后用户已批准将side70固定为默认；[新视角实验](SIDE70_PIPELINE.md)使用独立数据和训练权重。

P6000项目目录下的验收产物：

```text
experiments/p6000-expert-20261109/
  collection-02.log
  collection-02/
    collection-verification.json
    dwell-audit.json
    hdf5-audit.json
    physical-acceptance.jsonl
    expert-dual-camera.mp4
    robomimic_data/Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0-vis-teaser/experts.hdf5
```

## 可复现入口

资源和解释器填写在机器自己的 `configs/local.yaml` 中。P6000 设置：

```yaml
disabled_components:
  generate: GPT建模在5090-1执行；本机仅消费场景资产。
runtime_library_dirs:
  - /your/project/runtime/usr/lib/x86_64-linux-gnu
expert_cuda_arch: '12.0'
expert_build_jobs: 4
# 同时填写已有 Isaac 安装与固定版本 nvdiffrast 源码：
isaac_experience: /your/isaacsim/apps/isaacsim.exp.base.python.kit
upstreams:
  nvdiffrast: /your/project/upstream/nvdiffrast
```

先安装[环境说明](ENVIRONMENTS.md)中列出的独立环境与固定依赖，再构建扩展：

```bash
python scripts/build_expert_extensions.py --config /path/to/local.yaml --uv /path/to/uv
python scripts/build_expert_extensions.py --config /path/to/local.yaml --uv /path/to/uv --execute
python -m evoworld --config /path/to/local.yaml doctor --component collect
```

第一条仅打印构建计划；第二条才编译。源码不完整、CUDA工具链缺失时会停止。诊断通过不能替代真实采集验收。原始日志和模型权重保留在主机实验目录，不上传公开仓库。
