# P6000 专家轨迹环境：配置与验收进度

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

## 当前尚在进行

端到端燕麦奶任务采集验收：最多尝试3次，只保存1条成功示范，不启动SFT/RL训练。首次启动因上述依赖缺项退出；修复后的第二次运行已完成 Isaac 环境初始化并进入运动规划器创建阶段。

**在成功示范、HDF5内容与物理成功条件全部检查完成前，不宣称完整专家流水线已经通过验收。** 后续完成后更新本页。

## 可复现入口

资源和解释器填写在机器自己的 `configs/local.yaml` 中。P6000 设置：

```yaml
disabled_components:
  generate: GPT建模在5090-1执行；本机仅消费场景资产。
runtime_library_dirs:
  - /your/project/runtime/usr/lib/x86_64-linux-gnu
expert_cuda_arch: '12.0'
expert_build_jobs: 4
```

先安装[环境说明](ENVIRONMENTS.md)中列出的独立环境与固定依赖，再构建扩展：

```bash
python scripts/build_expert_extensions.py --config /path/to/local.yaml --uv /path/to/uv
python scripts/build_expert_extensions.py --config /path/to/local.yaml --uv /path/to/uv --execute
python -m evoworld --config /path/to/local.yaml doctor --component collect
```

第一条仅打印构建计划；第二条才编译。源码不完整、CUDA工具链缺失时会停止。诊断通过不能替代真实采集验收。原始日志和模型权重保留在主机实验目录，不上传公开仓库。
