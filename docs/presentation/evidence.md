# 展示素材与评测来源

2026-11-17 仅通过 SSH 读取 p5000pro 历史产物，未运行训练、仿真或评测，未修改服务器项目代码。

## 图片与视频

以下路径均相对 p5000pro 的 `/home/hello/lizk/vlas/EvoWorld/`：

| 仓库素材 | 原始路径 | 含义 |
| --- | --- | --- |
| media/main.jpg | inputs/phase5/main.jpg | 历史真实输入照片 |
| media/office_overview.png | remote/5090-1/experiments/phase8-office05-down-20261108-161923/office_overview.png | 历史生成场景预览 |
| media/tabletop_preview.png | remote/5090-1/experiments/phase8-office05-down-20261108-161923/tabletop_preview.png | 桌面任务预览，不作为 side70 相机版本证据 |
| media/episode-000-final.png | maintenance/dsrl-final-eval-20261110/episode-000-final.png | DSRL 成功回合末帧 |
| media/episode-000.mp4 | maintenance/dsrl-final-eval-20261110/results/episode-000.mp4 | seed 20000 成功视频，647 控制步 |

照片和场景分别来自历史阶段目录，不宣称是精确配对或未经人工修正的一次生成结果。图标是 imagegen 概念插画，与这些真实产物分开展示。

## 结果

`comparison.json`、`report.json` 原样复制自 `maintenance/dsrl-final-eval-20261110/results/`。前者含全部 20 个种子的配对结果；后者含 DSRL 逐回合物理判定。SFT 4/20 与 DSRL 9/20 来自该原始 comparison，且与 docs/SIDE70_PIPELINE.md 一致。

保持 side70、seeds 20000–20019、chunk10、horizon1000、denoise10。DSRL 第 290 轮，检查点哈希见原报告。20 个种子中 6 条新增成功、1 条原成功转失败、3 条共同成功、10 条共同失败。

SFT 成功率 20%；DSRL 成功率 45%，95% Wilson 区间约 25.8%–65.8%。不是广泛泛化或真机结果，两策略使用各自原生噪声推理方式，不能作为隔离训练因素的消融结论。

## 文档迁移

原中文 README 完整保留到 docs/usage-guide.md，旧英文完整保留到 docs/usage-guide.en.md 并增加 side70 状态更新提示。只调整相对链接，既有技术说明未删除。
