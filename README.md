# 基于扩展卡尔曼滤波的目标检测与运动轨迹预测系统

## 项目简介

本项目是一个完整的多目标跟踪（MOT）系统，以**扩展卡尔曼滤波（EKF）**为核心，结合 YOLOv8n 目标检测、CTRV 运动模型、匈牙利算法数据关联，实现对视频中多个目标的实时检测、状态估计、轨迹管理与未来轨迹预测。

本项目适合作为计算机视觉、目标跟踪方向的毕业设计项目，代码结构清晰、模块化程度高、注释详细，适合在普通笔记本（AMD CPU）上运行。

---

## 系统架构说明

```
视频输入
   ↓
帧采样 & 预处理
   ↓
目标检测（YOLOv8n / ONNX）
   ↓
测量向量提取
   ↓
EKF 预测步骤（CTRV 运动模型）
   ↓
数据关联（IoU + Mahalanobis + 匈牙利算法）
   ↓
EKF 更新步骤
   ↓
轨迹生命周期管理（Tentative → Confirmed → Lost → Removed）
   ↓
未来轨迹预测（1/5/10 帧递推）
   ↓
可视化（检测框 + 轨迹线 + 预测轨迹 + 协方差椭圆）
   ↓
输出（视频 + JSON + CSV）
```

---

## 目录结构说明

```
ekf_target_detection_prediction/
├── configs/          # 所有配置文件（YAML）
├── assets/           # 示例资源
├── scripts/          # 可执行脚本
├── src/ekf_mot/      # 核心代码包
│   ├── core/         # 类型、接口、配置、常量
│   ├── data/         # 视频读取、帧采样、数据变换
│   ├── detection/    # 目标检测模块
│   ├── filtering/    # EKF 滤波核心
│   ├── tracking/     # 多目标跟踪
│   ├── prediction/   # 轨迹预测
│   ├── metrics/      # 评估指标
│   ├── visualization/# 可视化
│   ├── serving/      # FastAPI 服务
│   └── utils/        # 工具函数
└── tests/            # 单元测试
```

---

## 环境安装方法

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：如果你的机器没有 NVIDIA GPU，`torch` 会自动使用 CPU 版本，无需额外配置。

### 3. 安装项目包（可选，用于开发模式）

```bash
pip install -e .
```

---

## 权重下载方法

运行以下脚本自动下载 YOLOv8n 权重：

```bash
python scripts/download_weights.py
```

权重将保存到 `weights/yolov8n.pt`。

---

## 如何运行 Demo

### 快速演示（使用示例视频）

```bash
python scripts/run_demo.py --config configs/exp/demo_cpu.yaml
```

### 指定视频文件

```bash
python scripts/run_demo.py --video path/to/your/video.mp4 --output outputs/result.mp4
```

---

## 如何运行跟踪

```bash
python scripts/run_tracking.py \
    --config configs/exp/demo_cpu.yaml \
    --video path/to/video.mp4 \
    --output outputs/
```

或使用主入口：

```bash
python -m src.ekf_mot.main --config configs/exp/demo_cpu.yaml --video path/to/video.mp4
```

---

## 如何导出 ONNX

```bash
python scripts/export_onnx.py --weights weights/yolov8n.pt --output weights/yolov8n.onnx
```

---

## 如何运行 API 服务

```bash
uvicorn src.ekf_mot.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

访问 API 文档：http://localhost:8000/docs

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /predict/frame | 单帧预测 |
| POST | /predict/video | 视频处理 |

---

## 配置文件说明

### 主要配置层级

1. `configs/base.yaml` - 全局基础配置
2. `configs/detector/yolov8n.yaml` - 检测器配置
3. `configs/tracker/ekf_ctrv.yaml` - EKF 跟踪器配置
4. `configs/exp/demo_cpu.yaml` - 实验配置（覆盖基础配置）

### 关键配置项

```yaml
# 检测器
detector:
  backend: ultralytics  # 或 onnx
  model: yolov8n
  conf: 0.35
  imgsz: 640

# 跟踪器
tracker:
  max_age: 20          # 最大丢失帧数
  n_init: 3            # 确认所需帧数
  dt: 0.04             # 时间步长（1/25s）

# 预测
prediction:
  future_steps: [1, 5, 10]  # 预测未来帧数

# 可视化
visualization:
  draw_future: true
  draw_covariance: true
```

---

## 评估说明

```bash
python scripts/evaluate.py \
    --pred outputs/tracks.json \
    --gt path/to/ground_truth.json \
    --output outputs/metrics.json
```

评估指标包括：
- 检测指标：Precision、Recall、F1
- 跟踪指标：MOTA、MOTP、ID Switch
- 预测指标：ADE、FDE、RMSE
- 运行时指标：FPS、平均帧耗时

---

## 冒烟测试

```bash
python scripts/smoke_test.py
```

此脚本会检查：
- Python 依赖是否安装
- 权重文件是否存在
- 配置文件是否可读
- 用伪造数据跑通完整流程

---

## 常见问题

### Q: 运行时提示 `No module named 'ultralytics'`
A: 运行 `pip install ultralytics`

### Q: 运行时提示 `No module named 'onnxruntime'`
A: 运行 `pip install onnxruntime`（CPU版本，无需GPU）

### Q: 视频处理速度很慢
A: 在配置中设置 `runtime.frame_skip: 2` 跳帧处理，或降低 `detector.imgsz` 到 320

### Q: 内存不足
A: 设置 `tracker.max_age: 10` 减少轨迹保留时间，设置 `runtime.max_frames` 限制处理帧数

### Q: 检测效果不好
A: 调整 `detector.conf` 阈值（降低可提高召回率），或切换到更大的模型如 yolov8s

### Q: 如何使用 ONNX 后端
A: 先导出 ONNX 模型，然后在配置中设置 `detector.backend: onnx`

---

## 后续扩展建议

1. **引入 ReID 模块**：在外观相似场景下提升 ID 一致性
2. **支持 ByteTrack 关联策略**：改进低分检测框的二阶段关联
3. **引入 StrongSORT**：结合卡尔曼滤波与外观特征
4. **支持 3D 目标跟踪**：扩展状态向量到三维空间
5. **引入 Transformer 预测器**：用 Transformer 替代 EKF 递推做轨迹预测
6. **支持多摄像头融合**：扩展到多视角目标跟踪
7. **在线学习**：根据跟踪结果动态调整噪声参数
8. **支持 MOT17/MOT20 标准评估**：接入 TrackEval 框架

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 目标检测 | YOLOv8n (Ultralytics / ONNX Runtime) |
| 状态估计 | 扩展卡尔曼滤波 (EKF) |
| 运动模型 | CTRV (恒转率恒速度) |
| 数据关联 | 匈牙利算法 + IoU + Mahalanobis |
| 轨迹预测 | EKF 递推 |
| 可视化 | OpenCV |
| API 服务 | FastAPI |
| 配置管理 | YAML + Pydantic |
| 测试框架 | pytest |

---

## 许可证

MIT License
