# 基于扩展卡尔曼滤波的目标检测与运动轨迹预测系统

## 项目简介

本项目是一个完整的多目标跟踪（MOT）系统，以**扩展卡尔曼滤波（EKF）**为核心，结合 YOLOv8n 目标检测、CTRV 运动模型、匈牙利算法三阶段数据关联，实现对视频中多个目标的实时检测、状态估计、轨迹管理与未来轨迹预测。

本项目适合作为计算机视觉、目标跟踪方向的毕业设计项目，代码结构清晰、模块化程度高、注释详细，适合在普通笔记本（AMD CPU）上运行。

---

## 系统架构

```
视频输入
   ↓
帧采样 & 预处理（frame_skip 跳帧，BGR→RGB）
   ↓
目标检测（YOLOv8n / ONNX 推理）
   ↓
测量向量提取（xyxy → cx, cy, w, h）
   ↓
EKF 预测步骤（CTRV 运动模型，7 维状态）
   ↓
三阶段数据关联（Stage A: IoU+Mahal+中心 / Stage B: IoU低分 / Stage C: Tentative）
   ↓
EKF 更新步骤（Joseph 形式协方差更新）
   ↓
轨迹生命周期管理（Tentative → Confirmed → Lost → Removed）
   ↓
未来轨迹预测（EKF 递推，1/5/10 帧）
   ↓
可视化（检测框 + 轨迹线 + 预测轨迹 + 协方差椭圆）
   ↓
输出（视频 H.264 + JSON + CSV）
```

---

## 目录结构

```
EKFsystem/
├── assets/
│   └── samples/
│       └── demo.mp4          # 示例演示视频
├── configs/
│   ├── base.yaml             # 全局基础配置
│   ├── paths.yaml            # 路径配置
│   ├── data/                 # 数据集配置（demo / mot17 / ua_detrac）
│   ├── detector/             # 检测器配置（yolov8n / yolov8_onnx）
│   ├── exp/                  # 实验配置（demo_cpu / person / vehicle / ua_detrac）
│   └── tracker/              # 跟踪器配置（ekf_ctrv / association / lifecycle）
├── docs/                     # 中文文档目录（毕业设计支撑文档）
│   ├── 01_background_and_significance.md
│   ├── 02_related_work.md
│   ├── 03_research_contents.md
│   ├── 04_thesis_outline.md
│   ├── 05_ekf_modeling.md
│   ├── 06_tracking_and_prediction.md
│   └── 07_experiment_setup.md
├── frontend/                 # React + TypeScript + Vite 前端
│   ├── src/
│   │   ├── pages/            # DashboardPage / UploadPredictPage / CameraPredictPage
│   │   ├── components/       # 公共组件（Layout、Navbar、LoadingSpinner 等）
│   │   ├── hooks/            # useCamera（摄像头管理）
│   │   ├── services/         # api.ts（axios 封装）、types.ts
│   │   └── store/            # Zustand 全局状态
│   ├── .env                  # VITE_API_BASE_URL=http://localhost:8000
│   └── package.json
├── scripts/
│   ├── run_demo.py           # 视频演示脚本
│   ├── run_tracking.py       # 跟踪脚本
│   ├── run_webcam.py         # 摄像头实时脚本
│   ├── evaluate.py           # 统一评估入口（检测/跟踪/预测三模式）
│   ├── smoke_test.py                # 环境冒烟测试
│   ├── convert_annotations.py      # 标注格式转换（UA-DETRAC / MOT17）
│   ├── validate_detector.py         # 检测器精度评估（P/R/F1/AP50）
│   ├── generate_demo_gt.py          # 生成演示伪 GT（用于检测评估演示）
│   ├── compare_baseline_vs_ekf.py   # Baseline vs EKF 轨迹质量对比
│   ├── run_experiments.py           # 批量实验（多配置×多视频，自动汇总）
│   ├── download_weights.py          # 权重下载
│   └── export_onnx.py               # ONNX 模型导出
├── src/ekf_mot/              # 核心 Python 包
│   ├── core/                 # 类型、接口、配置、常量
│   ├── data/                 # 视频读取、帧采样、格式转换
│   ├── detection/            # YOLOv8 检测器（ultralytics / ONNX 两后端）
│   ├── filtering/            # EKF 核心、CTRV 模型、雅可比矩阵、噪声矩阵
│   ├── metrics/              # 检测/跟踪/预测/运行时四类评估指标
│   ├── prediction/           # 轨迹预测器、平滑器、不确定性估计
│   ├── serving/              # FastAPI 接口（api.py / service.py / schemas.py）
│   ├── tracking/             # 多目标跟踪器、三阶段关联、轨迹状态机
│   ├── utils/                # 日志、计时、几何工具
│   ├── visualization/        # 检测框/轨迹/预测轨迹/协方差椭圆绘制
│   └── main.py               # 主入口
├── tests/                    # pytest 单元测试（EKF / CTRV / 关联 / 跟踪器）
├── weights/
│   └── yolov8n.pt            # YOLOv8n 预训练权重（运行 download_weights.py 下载）
├── outputs/                  # 处理结果输出目录（视频 / JSON / CSV）
├── requirements.txt
└── pyproject.toml
```

---

## 环境安装

### 1. Python 环境

```bash
# 创建并激活虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装项目包（可选，开发模式）
pip install -e .
```

> 无需 GPU，所有推理均在 CPU 上运行。

### 2. 下载 YOLOv8n 权重

```bash
python scripts/download_weights.py
# 权重保存至 weights/yolov8n.pt（约 6 MB）
```

### 3. 前端环境

```bash
cd frontend
npm install
```

---

## 运行方法

### 方式一：命令行 Demo

```bash
# 使用内置示例视频（assets/samples/demo.mp4）
python scripts/run_demo.py --config configs/exp/demo_cpu.yaml

# 指定自定义视频
python scripts/run_demo.py \
    --config configs/exp/demo_cpu.yaml \
    --video path/to/your/video.mp4
```

输出保存至 `outputs/demo/`，包含 `output_YYYYMMDD_HHMMSS.mp4`、`tracks_*.json`、`tracks_*.csv`。

### 方式二：Web 界面（后端 + 前端联调）

```bash
# 终端 1：启动后端（项目根目录执行）
uvicorn src.ekf_mot.serving.api:app --host 0.0.0.0 --port 8000 --reload

# 终端 2：启动前端
cd frontend
npm run dev
```

浏览器访问 **http://localhost:5173**，支持：
- **总控台**：后端连接状态监控
- **视频上传预测**：拖拽上传视频 → 异步处理 → 结果视频在线播放 + 下载
- **摄像头实时预测**：调用浏览器摄像头 → 逐帧发送 → Canvas 实时绘制检测框和预测轨迹

API 文档（Swagger UI）：http://localhost:8000/docs

### 方式三：摄像头命令行

```bash
python scripts/run_webcam.py --save-video
# 输出保存至 outputs/webcam_YYYYMMDD_HHMMSS.mp4
```

---

## API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/predict/frame` | 单帧预测（摄像头实时模式），请求体含 Base64 图像 |
| POST | `/predict/video/start` | 上传视频并启动异步处理任务，返回 `task_id` |
| GET | `/predict/video/status/{task_id}` | 轮询任务进度（status / progress / result） |
| GET | `/outputs/{filename}` | 获取处理完成的输出视频（H.264，浏览器可直接播放） |
| POST | `/reset` | 重置跟踪器状态 |

**视频处理异步流程**：

```
POST /predict/video/start → 返回 task_id
    → 每2秒 GET /predict/video/status/{task_id}（前端轮询）
    → status == "done" 后
    → GET /outputs/{result.output_file} 获取输出视频
```

> 注意：`/predict/video`（无 `/start` 后缀）为旧版同步接口，仅返回统计数据，不生成输出视频。

---

## 配置文件说明

配置采用分层覆盖：实验配置 > 基础配置。

```yaml
# configs/exp/demo_cpu.yaml（CPU 演示配置）

detector:
  backend: ultralytics    # 推理后端：ultralytics 或 onnx
  model: yolov8n
  conf: 0.35              # 检测置信度阈值
  classes: [0, 2]         # 检测类别：0=行人, 2=车辆

tracker:
  dt: 0.04               # 时间步长（秒），1/25s 对应 25fps
  auto_dt: true          # 从视频 FPS 自动估算 dt
  n_init: 3              # 轨迹确认所需连续命中帧数
  max_age: 15            # 轨迹最长存活帧数

prediction:
  future_steps: [1, 5, 10]  # 预测未来第 1/5/10 帧位置

visualization:
  draw_future: true
  draw_covariance: true
  track_history_len: 20

runtime:
  frame_skip: 2          # 每 2 帧处理 1 帧
  max_frames: 500        # 最多处理 500 帧（演示用）
```

---

## 数据准备与转换

### 支持的数据集格式

| 数据集 | 标注格式 | 转换状态 |
|--------|---------|---------|
| UA-DETRAC | XML（逐帧目标框） | ✅ 完全支持 |
| MOT17 | gt.txt（MOTC 格式） | ✅ 完全支持 |

转换输出为内部统一 JSON 格式，保存至 `data/processed/`。

### UA-DETRAC 转换

```bash
# 单个 XML 文件
python scripts/convert_annotations.py \
    --format ua_detrac \
    --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/MVI_20011.xml \
    --output data/processed/ua_detrac/ \
    --validate

# 批量转换整个目录
python scripts/convert_annotations.py \
    --format ua_detrac \
    --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/ \
    --output data/processed/ua_detrac/
```

### MOT17 转换

```bash
# 单个序列目录
python scripts/convert_annotations.py \
    --format mot17 \
    --input data/MOT17/train/MOT17-02-FRCNN \
    --output data/processed/mot17/

# 批量转换 train/ 目录
python scripts/convert_annotations.py \
    --format mot17 \
    --input data/MOT17/train/ \
    --output data/processed/mot17/
```

详细说明见 [docs/09_dataset_preparation.md](docs/09_dataset_preparation.md)。

---

## 检测评估

### 纯检测统计（无 GT）

```bash
python scripts/validate_detector.py \
    --video assets/samples/demo.mp4 \
    --output outputs/detection/
```

输出 `outputs/detection/detection_metrics.json`，包含：平均 FPS、每帧检测数等。

### 对照 GT 计算精度指标（需先转换标注）

```bash
python scripts/validate_detector.py \
    --video data/UA-DETRAC/video/MVI_20011.mp4 \
    --gt-json data/processed/ua_detrac/MVI_20011.json \
    --output outputs/detection/ \
    --iou-threshold 0.5
```

输出指标（`detection_metrics.json`）：

```json
{
  "global": {
    "precision": 0.78,
    "recall": 0.65,
    "f1": 0.71,
    "ap50": 0.68,
    "tp": 4821, "fp": 1342, "fn": 2543
  }
}
```

### 跟踪 / 预测评估（统一入口）

`scripts/evaluate.py` 支持四种模式，对照 GT JSON 计算完整指标：

```bash
# 跟踪评估（MOTA / MOTP / ID Switch）
python scripts/evaluate.py \
    --mode tracking \
    --pred outputs/demo/tracks_*.json \
    --gt   data/processed/demo_gt.json \
    --output outputs/metrics/tracking_eval.json

# 预测评估（ADE / FDE / RMSE + 逐步分解）
python scripts/evaluate.py \
    --mode prediction \
    --pred outputs/demo/tracks_*.json \
    --gt   data/processed/demo_gt.json \
    --output outputs/metrics/prediction_eval.json

# 全量评估（检测 + 跟踪 + 预测，一次性输出）
python scripts/evaluate.py \
    --mode all \
    --pred outputs/demo/tracks_*.json \
    --gt   data/processed/demo_gt.json \
    --output outputs/metrics/full_eval.json
```

- **检测指标**：Precision、Recall、F1、AP50（`src/ekf_mot/metrics/detection_metrics.py`）
- **跟踪指标**：MOTA、MOTP、ID Switch、平均轨迹长度（`src/ekf_mot/metrics/tracking_metrics.py`）
- **预测指标**：ADE、FDE、RMSE + 各步分解（`src/ekf_mot/metrics/prediction_metrics.py`）
- **运行时指标**：FPS、平均帧耗时（`src/ekf_mot/metrics/runtime_metrics.py`）

---

## 课题任务覆盖情况

| 需求项 | 当前状态 | 对应代码/文档位置 |
|--------|---------|-----------------|
| 目标检测模块（YOLOv8n） | ✅ 已实现 | `src/ekf_mot/detection/` |
| EKF 状态估计（CTRV） | ✅ 已实现 | `src/ekf_mot/filtering/ekf.py`，`models/ctrv.py` |
| 三阶段数据关联 | ✅ 已实现 | `src/ekf_mot/tracking/association.py` |
| 轨迹生命周期管理 | ✅ 已实现 | `src/ekf_mot/tracking/lifecycle.py` |
| 多步轨迹预测（1/5/10帧） | ✅ 已实现 | `src/ekf_mot/prediction/trajectory_predictor.py` |
| 可视化输出（视频+轨迹+预测轨迹） | ✅ 已实现 | `src/ekf_mot/visualization/` |
| JSON/CSV 结果输出 | ✅ 已实现 | `src/ekf_mot/main.py`，`serving/service.py` |
| FastAPI Web 服务（异步视频处理） | ✅ 已实现 | `src/ekf_mot/serving/api.py` |
| React 前端（上传+摄像头预测） | ✅ 已实现 | `frontend/` |
| 检测/跟踪/预测评估指标框架 | ✅ 已实现 | `src/ekf_mot/metrics/`（MOTA/MOTP/ADE/FDE/AP50 完整实现） |
| 统一评估入口（三模式） | ✅ 已实现 | `scripts/evaluate.py`（detection/tracking/prediction/all） |
| 配置文件系统（YAML 分层） | ✅ 已实现 | `configs/` |
| 单元测试覆盖 | ✅ 已实现 | `tests/`（8 个测试文件） |
| 中文文档（背景/EKF建模/实验环境等） | ✅ 已实现 | `docs/`（11 个文档，含结果分析模板） |
| 标准数据集接入（MOT17/UA-DETRAC） | ⚠️ 部分实现 | `configs/data/mot17.yaml`，`ua_detrac.yaml`（配置已有，数据需自行下载） |
| 逐帧定量评估（需标注数据） | ✅ 已实现 | `scripts/evaluate.py`（TrackingEvaluator 含帧级 IoU 匹配 + ID Switch） |
| Baseline 对比实验（IoU 关联 vs EKF） | ✅ 已实现 | `src/ekf_mot/prediction/baseline.py`，`scripts/compare_baseline_vs_ekf.py` |
| HOTA 等专业跟踪评估 | 🔲 待补齐 | 计划后续添加 |

---

## 实验与评估

### 快速冒烟测试

```bash
python scripts/smoke_test.py
```

验证：依赖是否安装完整、权重是否存在、配置是否可解析、流水线是否可跑通（用随机伪帧数据）。

### 单元测试

```bash
pytest tests/ -v
```

覆盖：EKF 预测/更新步、CTRV 雅可比矩阵、三阶段关联、轨迹状态机、端到端冒烟流水线。

### 实验场景配置

| 配置文件 | 场景 | 关键参数调整 | 输出目录 |
|---------|------|------------|---------|
| `scenario_uniform_motion.yaml` | 匀速运动 | `std_acc=0.5`，低噪声，长程预测 | `outputs/scenario_uniform/` |
| `scenario_accelerated_motion.yaml` | 变速运动 | `std_acc=5.0`，高加速度噪声 | `outputs/scenario_accelerated/` |
| `scenario_turning_motion.yaml` | 转弯运动 | `std_yaw_rate=1.5`，CTRV 主导 | `outputs/scenario_turning/` |
| `demo_person_accuracy.yaml` | 行人精度 | 底部中心锚点，宽高比过滤 | `outputs/person_accuracy/` |
| `demo_vehicle_accuracy.yaml` | 车辆精度 | 只检测 car/truck | `outputs/vehicle_accuracy/` |

```bash
# 单场景运行
python scripts/run_demo.py --config configs/exp/scenario_uniform_motion.yaml
python scripts/run_demo.py --config configs/exp/scenario_accelerated_motion.yaml
python scripts/run_demo.py --config configs/exp/scenario_turning_motion.yaml
```

### 批量实验

`scripts/run_experiments.py` 自动遍历配置文件 × 视频文件的所有组合，汇总结果到 JSON + CSV：

```bash
# 最简运行：用默认 demo 视频测试所有场景配置
python scripts/run_experiments.py --filter scenario --max-frames 300

# 指定配置目录和视频目录
python scripts/run_experiments.py \
    --config-dir configs/exp/ \
    --video-dir  assets/samples/ \
    --output-dir outputs/experiments/

# 只测试特定配置
python scripts/run_experiments.py \
    --configs configs/exp/scenario_uniform_motion.yaml \
             configs/exp/scenario_turning_motion.yaml \
    --video  assets/samples/demo.mp4
```

输出文件：

| 文件 | 内容 |
|------|------|
| `outputs/experiments/experiment_summary.json` | 完整实验结果（含所有指标） |
| `outputs/experiments/experiment_summary.csv` | CSV 格式（可直接用 Excel/pandas 绘图） |
| `outputs/experiments/fps_bar.png` | FPS 柱状图（需 matplotlib） |
| `outputs/experiments/track_length_bar.png` | 平均轨迹长度柱状图（需 matplotlib） |
| `outputs/experiments/<场景名>_<视频名>/` | 每次实验的输出视频 + 轨迹 JSON/CSV |

### 结果分析

按照 [docs/08_result_analysis_template.md](docs/08_result_analysis_template.md) 的模板，将 `experiment_summary.csv` 中的数值填入论文相应章节。

---

## Baseline 对比实验

对照实验：在同一视频上分别运行**纯 IoU 关联 Baseline**（无滤波）与 **EKF-CTRV 系统**，通过轨迹抖动、平滑度、平均长度等指标量化 EKF 的改善效果。

```bash
python scripts/compare_baseline_vs_ekf.py \
    --video assets/samples/demo.mp4 \
    --config configs/exp/demo_cpu.yaml \
    --output outputs/comparison/ \
    --max-frames 300
```

输出：
- `outputs/comparison/baseline_result.json` — Baseline 轨迹统计
- `outputs/comparison/ekf_result.json`      — EKF 轨迹统计
- `outputs/comparison/compare_summary.json` — 对比摘要（含改善百分比）

| 指标 | Baseline | EKF | 说明 |
|------|---------|-----|------|
| 轨迹抖动（Jitter） | 较高 | 较低 | EKF 抑制检测噪声 |
| 轨迹平滑度 | 较高 | 较低 | CTRV 模型建模转弯 |
| 平均轨迹长度 | 较短 | 较长 | Lost 状态机减少断轨 |

详见 [docs/10_baseline_vs_ekf.md](docs/10_baseline_vs_ekf.md)。

---

## 常见问题

### Q: 运行时提示 `No module named 'ultralytics'`
A: `pip install ultralytics`

### Q: 运行时提示 `No module named 'onnxruntime'`
A: `pip install onnxruntime`

### Q: 视频处理速度慢
A: 在配置中设置 `runtime.frame_skip: 2` 跳帧处理，或降低 `detector.imgsz` 到 320

### Q: 内存不足
A: 设置 `tracker.max_age: 10`，或设置 `runtime.max_frames` 限制处理帧数

### Q: 前端无法连接后端（显示"后端离线"）
A: 确认后端已启动（`uvicorn ... --port 8000`），检查 `frontend/.env` 中 `VITE_API_BASE_URL` 是否为 `http://localhost:8000`

### Q: 上传视频处理完后无法在浏览器播放
A: 系统已使用 `avc1`（H.264）编码，若仍无法播放，页面会显示下载按钮，下载后用本地播放器打开

### Q: 如何使用 ONNX 后端
A: 先导出 ONNX 模型（`python scripts/export_onnx.py`），然后在配置中设置 `detector.backend: onnx`

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 目标检测 | YOLOv8n（Ultralytics / ONNX Runtime） |
| 状态估计 | 扩展卡尔曼滤波（EKF），7 维状态 |
| 运动模型 | CTRV（恒转率恒速度），omega≈0 自动退化为 CV |
| 数据关联 | 三阶段匈牙利算法（IoU + Mahalanobis + 中心距离） |
| 轨迹预测 | EKF 多步递推，预测质量门限 |
| 可视化 | OpenCV |
| API 服务 | FastAPI + uvicorn（CORS 已配置） |
| 配置管理 | YAML + Pydantic v2 |
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS + Zustand |
| 测试框架 | pytest |

---

## 许可证

MIT License
