# 第八章 实验结果分析模板

> **使用说明**：本文档是论文"实验与结果分析"章节的写作模板，用 `[...]` 标记的内容需替换为真实数据。
> 运行 `scripts/run_experiments.py` 后从 `outputs/experiments/experiment_summary.csv` 读取数值填入。

---

## 8.1 实验环境

> 直接引用第七章（`docs/07_experiment_setup.md`）内容，此处简述。

| 项目 | 说明 |
|------|------|
| 处理器 | [AMD/Intel CPU 型号] |
| 内存 | [X] GB |
| 操作系统 | Windows 10/11 |
| Python 版本 | 3.11.x |
| 关键依赖 | ultralytics [版本]，numpy [版本]，scipy [版本] |
| 检测器 | YOLOv8n，权重 `yolov8n.pt`（约 6 MB） |
| 运行设备 | CPU（无 GPU） |

---

## 8.2 数据集说明

### 8.2.1 演示视频

| 项目 | 说明 |
|------|------|
| 文件名 | `assets/samples/demo.mp4` |
| 分辨率 | [宽×高] |
| 帧率 | [X] FPS |
| 时长 | [X] 秒（[X] 帧） |
| 内容 | [道路/行人/交叉口等场景描述] |
| 标注来源 | 由 `scripts/generate_demo_gt.py` 生成伪 GT（基于跟踪器 Confirmed 输出），用于评估流水线演示 |

> **注**：真实精度评估需使用 UA-DETRAC（详见 `docs/09_dataset_preparation.md`），受硬件与时间限制，本项目以演示视频为主要测试载体。

### 8.2.2 实验场景定义

| 场景 | 配置文件 | 运动特征 | 关键参数调整 |
|------|---------|---------|------------|
| 匀速运动 | `scenario_uniform_motion.yaml` | 近似匀速直线 | `std_acc=0.5`，`std_yaw_rate=0.1` |
| 变速运动 | `scenario_accelerated_motion.yaml` | 加速/减速 | `std_acc=5.0`，门控放宽 |
| 转弯运动 | `scenario_turning_motion.yaml` | 弧形曲线 | `std_yaw_rate=1.5`，`omega_threshold=0.005` |

---

## 8.3 评估指标体系

### 8.3.1 检测指标（`scripts/validate_detector.py`）

| 指标 | 含义 | 公式 |
|------|------|------|
| Precision (P) | 检测框中真实目标的比例 | TP / (TP + FP) |
| Recall (R) | 所有真实目标中被检出的比例 | TP / (TP + FN) |
| F1 | P 与 R 的调和均值 | 2PR / (P+R) |
| AP50 | IoU≥0.5 的平均精度（11点插值法） | 见 `detection_metrics.py` |

### 8.3.2 跟踪指标（`scripts/evaluate.py --mode tracking`）

| 指标 | 含义 | 公式 |
|------|------|------|
| MOTA | 多目标跟踪精度 | 1 - (FN + FP + IDSW) / GT |
| MOTP | 匹配对 IoU 均值（越高越好） | ΣIoU(match) / \|match\| |
| ID Switch | 同一目标在两帧间被不同 track_id 跟踪的次数 | — |
| avg_track_length | 轨迹平均持续帧数（反映连续性） | — |

### 8.3.3 预测指标（`scripts/evaluate.py --mode prediction`）

| 指标 | 含义 |
|------|------|
| ADE (px) | 所有预测步的平均位移误差 |
| FDE (px) | 最终预测步的位移误差 |
| RMSE (px) | 均方根误差 |
| per_step_ADE | 各预测步（step=1/5/10）单独的平均误差 |

### 8.3.4 轨迹质量指标（`scripts/compare_baseline_vs_ekf.py`）

| 指标 | 含义 |
|------|------|
| Jitter (px) | 帧间位移标准差（越小越稳定） |
| Smoothness (px) | 帧间加速度均值（越小越平滑） |

---

## 8.4 检测评估结果

> 运行命令：
> ```bash
> python scripts/generate_demo_gt.py
> python scripts/validate_detector.py \
>     --video assets/samples/demo.mp4 \
>     --gt-json data/processed/demo_gt.json \
>     --output outputs/detection/
> ```
> 结果文件：`outputs/detection/detection_metrics.json`

### 8.4.1 全局检测指标

| 指标 | 数值 |
|------|------|
| Precision | [从 detection_metrics.json 读取] |
| Recall | [从 detection_metrics.json 读取] |
| F1 | [从 detection_metrics.json 读取] |
| AP50 | [从 detection_metrics.json 读取] |
| 平均推理时间 | [X] ms/帧 |
| 平均 FPS | [X] FPS |

### 8.4.2 分类别指标

| 类别 | Precision | Recall | F1 | AP50 | TP | FP | FN |
|------|-----------|--------|----|------|-----|-----|-----|
| person (0) | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| car (2)    | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| truck (7)  | [X] | [X] | [X] | [X] | [X] | [X] | [X] |

### 8.4.3 分析

> 模板文字（根据实际数值调整）：
>
> YOLOv8n 在 demo 视频上的检测全局 Precision 为 [X]，Recall 为 [X]。
> 较低 Precision 主要来源于 [鸟类/小目标] 等误检，而 Recall 较高说明主要目标（车辆/行人）的漏检率低。
> 车辆类（car/truck）的 AP50 较高（[X]），因为车辆具有明显的外观特征；行人类 AP50 相对较低（[X]），原因是行人尺度小、遮挡频繁。

---

## 8.5 匀速场景分析

> 配置：`configs/exp/scenario_uniform_motion.yaml`
> 运行：`python scripts/run_demo.py --config configs/exp/scenario_uniform_motion.yaml`

### 8.5.1 运行时指标

| 指标 | 数值 |
|------|------|
| 处理帧数 | [X] |
| 平均 FPS | [X] |
| 检测轨迹数 | [X] |
| 平均轨迹长度 | [X] 帧 |

### 8.5.2 跟踪与预测指标

| 指标 | 数值 | 说明 |
|------|------|------|
| MOTA | [X] | 综合跟踪精度 |
| MOTP | [X] | 匹配框 IoU 均值 |
| ID Switch | [X] | 越小越好 |
| ADE (step=1) | [X] px | 1 步预测误差 |
| ADE (step=5) | [X] px | 5 步预测误差 |
| ADE (step=10) | [X] px | 10 步预测误差 |
| Jitter | [X] px | 轨迹抖动（vs Baseline: [X]） |

### 8.5.3 分析

> 匀速场景设置了低过程噪声（`std_acc=0.5`，`std_yaw_rate=0.1`），使 EKF 更倾向于信任 CTRV 运动预测。
> 在目标匀速行驶时，卡尔曼增益较小，状态更新幅度小，轨迹曲线平滑。
> 预测精度 ADE(1步)=[X] px 远优于 Baseline 线性外推（[X] px），因为目标真实运动符合 CTRV 假设。

---

## 8.6 变速场景分析

> 配置：`configs/exp/scenario_accelerated_motion.yaml`

### 8.6.1 运行时指标

| 指标 | 数值 |
|------|------|
| 处理帧数 | [X] |
| 平均 FPS | [X] |
| 检测轨迹数 | [X] |
| 平均轨迹长度 | [X] 帧 |

### 8.6.2 跟踪与预测指标

| 指标 | 均速场景 | 变速场景 | 变化说明 |
|------|---------|---------|---------|
| ADE (step=1) | [X] px | [X] px | 变速时误差增大 |
| ADE (step=10) | [X] px | [X] px | 长程预测受速度突变影响更大 |
| ID Switch | [X] | [X] | 变速导致关联难度上升 |
| Jitter | [X] px | [X] px | 变速轨迹抖动增大 |

### 8.6.3 分析

> 变速场景设置了高加速度噪声（`std_acc=5.0`），允许 EKF 状态向量中的速度分量快速调整。
> 当目标突然减速或加速时，卡尔曼增益增大（预测置信度下降），状态更新更多依赖检测结果。
> 预测误差 ADE 相比匀速场景增大约 [X]%，符合预期（速度突变无法被线性模型预测）。
> 协方差椭圆在速度变化后会短暂增大，随着新检测到来重新收敛。

---

## 8.7 转弯场景分析

> 配置：`configs/exp/scenario_turning_motion.yaml`

### 8.7.1 CTRV 模型 vs 线性 Baseline 对比

> 运行：`python scripts/compare_baseline_vs_ekf.py --config configs/exp/scenario_turning_motion.yaml`

| 指标 | Baseline（线性外推） | EKF-CTRV | 改善幅度 |
|------|---------------------|---------|---------|
| 轨迹抖动 Jitter (px) | [X] | [X] | [+X%] |
| 轨迹平滑度 (px) | [X] | [X] | [+X%] |
| 平均轨迹长度 | [X] 帧 | [X] 帧 | [+X%] |

### 8.7.2 分析

> 转弯场景是 CTRV 模型的核心优势场景。
> Baseline 线性外推假设目标匀速直线运动，在转弯时预测点沿切线方向飞出，ADE 急剧增大。
> EKF-CTRV 通过角速度状态 ω 维护目标的旋转信息，预测点沿弧形轨迹延伸，误差显著更小。
>
> 具体表现：
> - EKF 轨迹抖动相比 Baseline 降低约 [X]%（归因于测量噪声的平滑）
> - EKF 平均轨迹长度比 Baseline 长约 [X] 帧（归因于 Lost 状态在目标遮挡时保持轨迹）
> - 协方差椭圆在转弯方向上有明显延伸，说明 EKF 正确传播了位置不确定性

---

## 8.8 Baseline vs EKF 全面对比

> 数据来源：`outputs/comparison/compare_summary.json`
> 运行：`python scripts/compare_baseline_vs_ekf.py --max-frames 300`

### 8.8.1 汇总对比表

| 指标 | Baseline | EKF-CTRV | 改善 |
|------|---------|---------|-----|
| 轨迹抖动（px） | [X] | [X] | [+X%] |
| 轨迹平滑度（px） | [X] | [X] | [+X%] |
| 平均轨迹长度（帧） | [X] | [X] | [+X%] |
| 轨迹总数 | [X] | [X] | — |

### 8.8.2 结论

> EKF-CTRV 相比纯 IoU 关联 Baseline，在以下三个维度取得显著改善：
>
> 1. **轨迹稳定性**：EKF 滤波将帧间位移标准差降低 [X]%，说明卡尔曼增益有效平滑了检测器噪声
> 2. **轨迹平滑度**：帧间加速度均值降低 [X]%，说明 CTRV 运动模型比纯观测更准确反映目标运动
> 3. **轨迹连续性**：平均长度提升 [X]%，说明 Lost 状态机（max_age=15）在目标短暂遮挡时有效保持轨迹

---

## 8.9 系统不足与改进方向

### 8.9.1 当前局限

| 不足点 | 描述 | 影响 |
|--------|------|------|
| 评估数据有限 | 使用伪 GT（跟踪器自身输出），非人工标注 | MOTA/AP50 等指标偏乐观，不具备标准数据集可比性 |
| 检测器未调优 | YOLOv8n 使用原始预训练权重，未在本数据集微调 | 低 Precision（误检多）影响后端跟踪质量 |
| CPU 推理速度 | 平均 FPS 仅约 [X] FPS（CPU），无法实时 | 不适合对帧率要求高的场景 |
| 单摄像头视角 | 遮挡频繁时 ID Switch 增加 | 复杂遮挡场景跟踪精度下降 |
| 运动模型局限 | CTRV 假设恒定转率，在急弯场景仍有误差 | 大曲率转弯的长程预测精度受限 |

### 8.9.2 改进方向

| 方向 | 具体方法 | 预期收益 |
|------|---------|---------|
| 检测器微调 | 在目标域数据集上 fine-tune YOLOv8n | 降低 FP 数量，提升 Precision |
| 引入重识别特征 | 加入 ReID 特征到关联代价（如 DeepSORT） | 减少遮挡后的 ID Switch |
| GPU 加速 | 使用 CUDA 推理 | 提升 FPS 至 50+ |
| 更高阶运动模型 | 引入 IMM（Interacting Multiple Model）自适应切换运动模型 | 处理突变机动 |
| 标准测试集 | 在 UA-DETRAC 完整测试集上评估 | 获得具备发表价值的基准结果 |
| HOTA 指标 | 集成 TrackEval 框架计算 HOTA | 更全面的跟踪评估（同时衡量检测与关联） |

---

## 8.10 附录：实验运行命令速查

```bash
# 1. 生成伪 GT（首次运行前执行）
python scripts/generate_demo_gt.py

# 2. 检测评估
python scripts/validate_detector.py \
    --video assets/samples/demo.mp4 \
    --gt-json data/processed/demo_gt.json \
    --output outputs/detection/

# 3. 单场景实验
python scripts/run_demo.py --config configs/exp/scenario_uniform_motion.yaml
python scripts/run_demo.py --config configs/exp/scenario_accelerated_motion.yaml
python scripts/run_demo.py --config configs/exp/scenario_turning_motion.yaml

# 4. 批量实验（所有场景配置 × 所有视频）
python scripts/run_experiments.py \
    --filter scenario \
    --output-dir outputs/experiments/ \
    --max-frames 300

# 5. 跟踪/预测评估
python scripts/evaluate.py --mode all \
    --pred outputs/scenario_uniform/tracks.json \
    --gt   data/processed/demo_gt.json \
    --output outputs/metrics/uniform_eval.json

# 6. Baseline vs EKF 对比
python scripts/compare_baseline_vs_ekf.py \
    --config configs/exp/scenario_turning_motion.yaml \
    --max-frames 300 \
    --output outputs/comparison/

# 7. 查看批量实验汇总
cat outputs/experiments/experiment_summary.json
```
