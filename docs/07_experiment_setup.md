# 第七章 实验环境配置

> 本文档可直接用作论文"实验环境"章节初稿。

---

## 7.1 硬件环境

| 项目 | 说明 |
|------|------|
| 处理器 | AMD CPU（本项目全程在 CPU 上运行，无需 GPU） |
| 内存 | 建议 8 GB 及以上 |
| 存储 | 项目源码约 50 MB；权重文件 yolov8n.pt 约 6 MB；处理后视频按实际时长而定 |
| 显卡 | 不要求，系统自动回退到 CPU 模式 |

> 如有 NVIDIA GPU 且已安装 CUDA，ultralytics 会自动检测并使用 GPU 加速检测推理，无需修改代码。

---

## 7.2 操作系统

| 项目 | 说明 |
|------|------|
| 主测试环境 | Windows 10 / Windows 11 |
| 视频编解码 | OpenCV `avc1` (H.264) 通过 Windows Media Foundation 编码，浏览器可直接播放 |
| 路径分隔符 | 代码使用 `pathlib.Path`，兼容 Windows / Linux / macOS |

---

## 7.3 软件环境

### 7.3.1 Python 环境

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 建议使用虚拟环境（venv 或 conda） |
| numpy | >=1.24.0, <2.0.0 | EKF 矩阵运算 |
| scipy | >=1.10.0 | 匈牙利算法（`linear_sum_assignment`） |
| opencv-python | >=4.8.0 | 视频读写、图像处理 |
| ultralytics | >=8.0.0 | YOLOv8n 推理 |
| onnxruntime | >=1.16.0 | ONNX 推理后端（可选） |
| torch | >=2.0.0 | ultralytics 依赖（CPU 版即可） |
| fastapi | >=0.104.0 | Web API 服务框架 |
| uvicorn | >=0.24.0 | ASGI 服务器 |
| pydantic | >=2.0.0 | 数据校验与配置管理 |
| pandas | >=2.0.0 | 结果 CSV 处理 |
| pytest | >=7.4.0 | 单元测试框架 |

完整依赖见 `requirements.txt`。

### 7.3.2 Node.js 环境（前端）

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| Node.js | >=18.0.0 | 前端构建运行环境 |
| npm | >=9.0.0 | 包管理器 |
| React | 18.2.0 | 前端框架 |
| TypeScript | 5.x | 类型安全 |
| Vite | 5.x | 前端构建工具 |
| Tailwind CSS | 3.x | 样式框架 |
| Zustand | 4.x | 状态管理 |
| Axios | 1.x | HTTP 客户端 |

---

## 7.4 环境安装步骤

### 7.4.1 Python 后端环境

```bash
# Step 1：创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Step 2：安装依赖
pip install -r requirements.txt

# Step 3：安装项目包（开发模式，可选）
pip install -e .

# Step 4：下载 YOLOv8n 权重
python scripts/download_weights.py
# 权重保存至 weights/yolov8n.pt（约 6 MB）
```

### 7.4.2 前端环境

```bash
cd frontend
npm install          # 安装 node_modules（首次需要，后续可跳过）
```

---

## 7.5 运行方式

### 7.5.1 命令行 Demo 模式

```bash
# 使用示例视频（assets/samples/demo.mp4），CPU 配置
python scripts/run_demo.py --config configs/exp/demo_cpu.yaml

# 指定自定义视频
python scripts/run_demo.py \
    --config configs/exp/demo_cpu.yaml \
    --video path/to/your/video.mp4
```

输出保存到 `outputs/demo/`，包含 `output_YYYYMMDD_HHMMSS.mp4`（视频）、`tracks_*.json`（逐帧跟踪结果）、`tracks_*.csv`（汇总数据）。

### 7.5.2 Web 服务模式（后端 + 前端）

```bash
# 终端 1：启动后端
uvicorn src.ekf_mot.serving.api:app --host 0.0.0.0 --port 8000 --reload

# 终端 2：启动前端
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173` 进入 Web 界面。
API 文档（Swagger UI）：`http://localhost:8000/docs`

### 7.5.3 摄像头实时模式（命令行）

```bash
python scripts/run_webcam.py --save-video
# 输出保存至 outputs/webcam_YYYYMMDD_HHMMSS.mp4
```

---

## 7.6 配置文件说明

项目使用分层 YAML 配置，实验配置文件会覆盖基础配置的对应项：

| 配置文件 | 用途 |
|---------|------|
| `configs/base.yaml` | 全局基础配置，包含所有参数的默认值 |
| `configs/detector/yolov8n.yaml` | YOLOv8n 检测器默认配置 |
| `configs/tracker/ekf_ctrv.yaml` | EKF + CTRV 跟踪器配置（状态维度、噪声参数等） |
| `configs/tracker/association.yaml` | 数据关联参数（各阶段阈值、权重） |
| `configs/tracker/lifecycle.yaml` | 轨迹生命周期参数（n_init、max_age） |
| `configs/exp/demo_cpu.yaml` | CPU 演示配置（跳帧=2，最多500帧，只检测行人和车辆） |
| `configs/exp/demo_person_accuracy.yaml` | 行人精度实验配置 |
| `configs/exp/demo_vehicle_accuracy.yaml` | 车辆精度实验配置 |
| `configs/exp/ua_detrac_main.yaml` | UA-DETRAC 数据集实验配置 |

**关键参数说明**：

```yaml
# 检测器（configs/exp/demo_cpu.yaml）
detector:
  backend: ultralytics    # 推理后端：ultralytics 或 onnx
  model: yolov8n
  conf: 0.35              # 检测置信度阈值
  classes: [0, 2]         # 检测类别：0=行人, 2=车辆

# 跟踪器
tracker:
  dt: 0.04               # 时间步长（秒），1/25s 对应 25fps
  auto_dt: true          # 从视频 FPS 自动估算 dt
  n_init: 3              # 轨迹确认所需连续命中帧数
  max_age: 15            # 轨迹最长存活帧数（无匹配时）

# 预测
prediction:
  future_steps: [1, 5, 10]  # 预测未来第 1/5/10 帧的位置

# 可视化
visualization:
  draw_future: true      # 是否绘制预测轨迹
  draw_covariance: true  # 是否绘制协方差椭圆
  track_history_len: 20  # 轨迹线保留帧数
```

---

## 7.7 冒烟测试（快速验证环境）

```bash
python scripts/smoke_test.py
```

该脚本依次检查：
- Python 依赖包是否安装完整
- YOLOv8n 权重文件是否存在（`weights/yolov8n.pt`）
- 配置文件是否可正常解析
- 用随机生成的伪帧数据跑通一次完整检测+跟踪+预测流水线

全部通过则输出 `✓ 所有检查通过`，说明环境配置正确。

---

## 7.8 单元测试

```bash
pytest tests/ -v
```

测试覆盖：

| 测试文件 | 测试内容 |
|---------|---------|
| `test_ekf.py` | EKF 预测步、更新步、数值稳定性 |
| `test_ctrv.py` | CTRV 状态转移、雅可比矩阵 |
| `test_association.py` | 三阶段关联逻辑 |
| `test_track_manager.py` | 轨迹增删、生命周期转换 |
| `test_improvements.py` | Bootstrap 初始化、置信度自适应 R |
| `test_smoke_pipeline.py` | 端到端管道冒烟测试 |
| `test_detection_metrics.py` | IoU 计算、TP/FP/FN 匹配、Precision/Recall/F1/AP50 |
| `test_convert_annotations.py` | UA-DETRAC XML 解析、MOT17 gt.txt 解析、格式验证 |

---

## 7.9 数据准备与检测评估

### 标注格式转换

```bash
# UA-DETRAC XML → 内部 JSON
python scripts/convert_annotations.py \
    --format ua_detrac \
    --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/ \
    --output data/processed/ua_detrac/ \
    --validate

# MOT17 gt.txt → 内部 JSON
python scripts/convert_annotations.py \
    --format mot17 \
    --input data/MOT17/train/ \
    --output data/processed/mot17/ \
    --validate
```

### 检测器精度评估

```bash
# 模式1：纯检测统计（无标注数据，只统计检测数量和速度）
python scripts/validate_detector.py \
    --video assets/samples/demo.mp4 \
    --output outputs/detection/

# 模式2：对照 GT 标注计算 Precision/Recall/F1/AP50
python scripts/validate_detector.py \
    --gt-json data/processed/ua_detrac/MVI_20011.json \
    --video path/to/MVI_20011.mp4 \
    --output outputs/detection/ \
    --iou-threshold 0.5
```

### 评估报告格式

```json
{
  "iou_threshold": 0.5,
  "num_frames": 664,
  "global": {
    "precision": 0.7823,
    "recall": 0.6541,
    "f1": 0.7124,
    "ap50": 0.6890,
    "tp": 4821, "fp": 1342, "fn": 2543
  },
  "per_class": {
    "0": {"class_name": "car", "precision": 0.80, "recall": 0.67, ...}
  }
}
```

详见 [docs/09_dataset_preparation.md](09_dataset_preparation.md)。
