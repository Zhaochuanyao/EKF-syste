# EKF 车辆多目标跟踪与轨迹预测系统

基于扩展卡尔曼滤波（EKF）的车辆多目标跟踪与短时轨迹预测系统，面向交通监控场景。

核心采用 CTRV 运动模型 + 三阶段匈牙利关联，并实现了基于新息统计的 R/Q 自适应噪声调节机制作为创新点。系统提供 FastAPI 后端服务与 React 前端界面，支持视频文件批量处理和摄像头实时推理。

---

## 目录

- [系统架构](#系统架构)
- [核心算法](#核心算法)
- [创新点：自适应噪声调节](#创新点自适应噪声调节)
- [环境安装](#环境安装)
- [快速启动](#快速启动)
- [前端界面](#前端界面)
- [API 接口](#api-接口)
- [配置文件](#配置文件)
- [消融实验](#消融实验)
- [评估指标](#评估指标)
- [项目结构](#项目结构)

---

## 系统架构

```
输入（视频帧 / 摄像头帧）
        │
        ▼
  YOLOv8 检测器
  车辆类别过滤（car / truck / bus / motorcycle）
        │
        ▼
  三阶段匈牙利关联
  Stage A: Confirmed+Lost × 高置信检测（IoU+Mahal+中心距融合代价）
  Stage B: 未匹配轨迹 × 低置信检测（IoU 宽松匹配）
  Stage C: Tentative × 未匹配高置信检测（IoU 轻量匹配）
        │
        ▼
  CTRV-EKF 状态估计
  状态向量: [cx, cy, v, θ, ω, w, h]
  自适应噪声调节（可选）
        │
        ▼
  轨迹预测（未来 1/3/5/7/10 帧）
  预测质量门控 + Fixed-lag Smoothing
        │
        ▼
  可视化输出（检测框 / 轨迹线 / 预测点）
```

---

## 核心算法

### CTRV 运动模型

状态向量为 7 维：

```
x = [cx, cy, v, θ, ω, w, h]
    中心坐标  速度  航向角  角速度  宽高
```

观测向量为 4 维：`z = [cx, cy, w, h]`

非线性状态转移（`|ω| ≥ ε` 时）：

```
cx' = cx + (v/ω)(sin(θ + ω·dt) - sin(θ))
cy' = cy - (v/ω)(cos(θ + ω·dt) - cos(θ))
```

当 `|ω| < ε` 时退化为 CV 匀速直线模型，避免除零。

### 三阶段数据关联

| 阶段 | 轨迹集合 | 检测集合 | 代价函数 |
|------|---------|---------|---------|
| Stage A | Confirmed + Lost | 高置信（≥0.5） | IoU + Mahalanobis + 中心距 |
| Stage B | 未匹配 Confirmed + Lost | 低置信（≥0.15） | IoU |
| Stage C | Tentative | Stage A 未匹配高置信 | IoU |

Mahalanobis 距离使用 EKF 预测协方差 P 进行门控，阈值对应 χ²(0.999, df=4)。

### 轨迹生命周期

- **Tentative**：新建轨迹，连续命中 `n_init=2` 帧后升为 Confirmed
- **Confirmed**：稳定跟踪，参与预测输出
- **Lost**：连续未匹配，保留至 `max_age=40` 帧后删除
- **Deleted**：已删除，不再参与关联

---

## 创新点：自适应噪声调节

基于新息统计（NIS, Normalized Innovation Squared）的分层 R/Q 自适应噪声调度，在标准 CTRV-EKF 基础上提升复杂交通场景的鲁棒性。

### 原理

**NIS 定义：**

```
NIS = ν^T · S^{-1} · ν
```

其中 `ν = z - h(x̂)` 为新息向量，`S = H·P·H^T + R` 为新息协方差。

NIS 服从 χ²(df=4) 分布，超出阈值（9.4877，对应 p=0.99）表示观测异常。

### 五种策略

| 策略 | 标识 | 说明 |
|------|------|------|
| Current EKF | `current_ekf` | 固定 R/Q，标准 EKF，无自适应 |
| +R-adapt | `r_adapt` | 仅启用测量噪声 R 在线自适应 |
| +Q-sched | `q_sched` | 仅启用过程噪声 Q 机动感知调度 |
| +RQ-adapt | `rq_adapt` | R 自适应 + Q 调度，不启用鲁棒兜底 |
| Full Adpt | `full_adpt` | R/Q 自适应 + 异常观测鲁棒处理（完整模式） |

### R 自适应

观测异常时（NIS > `nis_threshold`），动态放大 R：

```
R_new = R_base · (1 + λ_R · δ̄²)
```

其中 `δ̄` 为新息偏差的指数移动平均，`λ_R=0.3`，`β=0.85`。

异常消除后，R 按 `recover_alpha_r=0.65` 衰减回基础值。

### Q 机动调度

机动得分综合 NIS、角速度变化、航向角速率三项：

```
maneuver_score = w_nis · clip(NIS/thr, cap) + w_ω · |Δω| + w_θ · |θ̇|
Q_scale = 1 + λ_Q · maneuver_score  （上限 q_max_scale=4.0）
```

### 鲁棒更新（Full Adpt 专属）

两级处理极端异常观测：

1. **Skip update**：NIS > `drop_threshold=20.0` 且检测分数 < `low_score=0.35` 时，跳过本帧 EKF 更新，保留预测结果
2. **Robust clip**：未触发 skip 时，对新息向量逐元素裁剪至 `±clip_delta=25.0` 像素，抑制离群观测冲击

---

## 环境安装

**Python 版本要求：** ≥ 3.9

```bash
# 克隆项目
git clone <repo-url>
cd EKFsystem

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装为可编辑包（可选，使 src 路径可直接导入）
pip install -e .
```

**主要依赖：**

| 包 | 用途 |
|----|------|
| `ultralytics` | YOLOv8 检测器 |
| `numpy / scipy` | EKF 矩阵运算、匈牙利算法 |
| `opencv-python` | 视频读写、可视化 |
| `fastapi / uvicorn` | 后端 API 服务 |
| `pydantic` | 数据校验 |

---

## 快速启动

### 启动后端服务

```bash
uvicorn src.ekf_mot.serving.api:app --host 0.0.0.0 --port 8000
```

服务启动后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 命令行运行（视频文件）

```bash
# 车辆场景（默认）
python scripts/run_demo.py --input path/to/video.mp4

# 指定配置
python scripts/run_demo.py --input video.mp4 --config demo_vehicle_accuracy

# 摄像头实时
python scripts/run_webcam.py
```

### 环境冒烟测试

```bash
python scripts/smoke_test.py
```

---

## 前端界面

前端基于 React 18 + TypeScript + Vite + Tailwind CSS，提供三个页面：

- **总控台**：系统入口、能力说明、创新点展示
- **视频预测**：上传视频文件，批量处理，下载结果视频
- **摄像头预测**：实时摄像头推理，画面叠加检测框与轨迹

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`（需后端服务已启动）。

### 构建生产版本

```bash
cd frontend
npm run build
```

### EKF 噪声策略切换

前端支持在运行时热切换五种 EKF 噪声策略，切换后下一帧立即生效，无需重启服务。

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/predict/frame` | 单帧推理（摄像头模式） |
| POST | `/predict/video/start` | 启动异步视频处理任务 |
| GET | `/predict/video/status/{task_id}` | 查询任务进度 |
| GET | `/outputs/{filename}` | 获取输出视频文件 |
| GET | `/api/ekf/noise-mode` | 获取当前噪声策略 |
| PUT | `/api/ekf/noise-mode` | 切换噪声策略 |
| POST | `/reset` | 重置跟踪器状态 |

### 单帧推理请求示例

```json
POST /predict/frame
{
  "image_base64": "<base64编码的JPEG图像>",
  "frame_id": 0,
  "config_name": "demo_vehicle_accuracy"
}
```

### 切换噪声策略

```json
PUT /api/ekf/noise-mode
{
  "mode": "full_adpt"
}
```

合法 mode 值：`current_ekf` / `r_adapt` / `q_sched` / `rq_adapt` / `full_adpt`

---

## 配置文件

配置文件位于 `configs/`，采用 YAML 格式，支持通过 `_base_` 字段继承基础配置。

### 预置实验配置

| 配置文件 | 说明 |
|---------|------|
| `configs/exp/demo_vehicle_accuracy.yaml` | 车辆场景（默认，推荐） |
| `configs/exp/demo_person_accuracy.yaml` | 行人场景 |
| `configs/exp/demo_cpu.yaml` | CPU 轻量模式 |
| `configs/exp/ua_detrac_main.yaml` | UA-DETRAC 数据集评估 |
| `configs/exp/uadetrac_adaptive_full.yaml` | UA-DETRAC + 完整自适应 |

### 关键配置项（以 `demo_vehicle_accuracy.yaml` 为例）

```yaml
detector:
  classes: [2, 3, 5, 7]       # car / motorcycle / bus / truck
  conf: 0.4                    # 检测置信度阈值

tracker:
  n_init: 2                    # 确认帧数
  max_age: 40                  # 最大丢失帧数
  gating_threshold_confirmed: 18.47   # Mahalanobis 门控阈值

ekf:
  process_noise:
    std_acc: 1.0               # 加速度噪声
    std_yaw_rate: 0.4          # 角速度噪声
  measurement_noise:
    std_cx: 6.0                # 检测中心 x 噪声（像素）
    std_cy: 6.0
    std_w: 10.0
    std_h: 10.0
    score_adaptive: true       # 根据检测分数自适应调整 R

prediction:
  future_steps: [1, 3, 5, 7, 10]     # 预测未来帧数
  fixed_lag_smoothing: true           # 固定滞后平滑
  smoothing_lag: 5
```

---

## 消融实验

六组消融实验对比不同噪声策略的跟踪性能：

| 组别 | 名称 | 说明 |
|------|------|------|
| G1 | BaselineTracker | 纯 IoU 关联 + 恒速外推，无 EKF |
| G2 | Current EKF | CTRV-EKF，无自适应噪声 |
| G3 | EKF+R-adapt | CTRV-EKF + 仅 R 自适应 |
| G4 | EKF+Q-schedule | CTRV-EKF + 仅 Q 机动调度 |
| G5 | EKF+RQ-adapt | CTRV-EKF + R+Q 自适应，无鲁棒更新 |
| G6 | Full Adaptive EKF | CTRV-EKF + R+Q 自适应 + 鲁棒更新 |

### 运行消融实验

```bash
# 使用合成序列（无需外部数据集）
python scripts/run_adaptive_ablation.py --data synthetic --sequences 8

# 使用 UA-DETRAC 数据集（需本地数据）
python scripts/run_adaptive_ablation.py --data uadetrac
```

输出文件保存至 `outputs/adaptive_ekf/uadetrac_subset/`：

| 文件 | 内容 |
|------|------|
| `main_metrics.csv` | 6 组跨序列平均指标（MOTA/MOTP/IDSW/AvgLen） |
| `ablation_metrics.csv` | 6 组 × N 序列逐序列原始指标 |
| `significance.csv` | G6 vs 其余 5 组 Wilcoxon 显著性检验 |
| `diagnostics.csv` | 自适应组（G3-G6）逐序列诊断统计 |

---

## 评估指标

### 跟踪指标

| 指标 | 说明 |
|------|------|
| MOTA | 多目标跟踪精度，综合 FP/FN/IDSW |
| MOTP | 多目标跟踪精度，平均定位误差 |
| ID Switch (IDSW) | 身份切换次数 |
| AvgLen | 平均轨迹长度（帧） |

### 自适应诊断指标

| 指标 | 说明 |
|------|------|
| skip_rate | 跳过更新帧占比（鲁棒更新触发率） |
| abnormal_rate | 异常观测帧占比（NIS 超阈值率） |
| avg_nis_ema | 平均 NIS 指数移动均值 |
| avg_maneuver | 平均机动得分 |

### 运行评估脚本

```bash
# 统一评估入口
python scripts/evaluate.py --config configs/exp/demo_vehicle_accuracy.yaml

# 检测器精度评估
python scripts/validate_detector.py

# EKF vs Baseline 对比
python scripts/compare_baseline_vs_ekf.py
```

---

## 项目结构

```
EKFsystem/
├── src/ekf_mot/
│   ├── core/           # 配置加载、类型定义、常量
│   ├── data/           # 视频读取、帧采样、数据转换
│   ├── detection/      # YOLOv8 检测器（Ultralytics / ONNX）
│   ├── filtering/      # EKF 核心、CTRV 模型、自适应噪声、鲁棒更新
│   │   ├── ekf.py              # ExtendedKalmanFilter
│   │   ├── adaptive_noise.py   # AdaptiveNoiseController
│   │   ├── robust_update.py    # skip update / robust clip
│   │   └── models/
│   │       ├── ctrv.py         # CTRV 状态转移与雅可比
│   │       └── cv.py           # CV 对照模型
│   ├── tracking/       # 多目标跟踪器、三阶段关联、轨迹生命周期
│   ├── prediction/     # 轨迹预测、质量门控、Fixed-lag Smoothing
│   ├── metrics/        # MOTA/MOTP/IDSW 评估
│   ├── visualization/  # 检测框、轨迹线、预测点绘制
│   ├── serving/        # FastAPI 服务（api.py / service.py）
│   └── config/         # 自适应策略枚举（adaptive_mode.py）
├── configs/
│   ├── base.yaml
│   ├── exp/            # 实验配置（vehicle / person / ua_detrac 等）
│   ├── tracker/        # EKF 参数、关联参数、生命周期参数
│   └── detector/       # 检测器配置
├── frontend/           # React + TypeScript + Vite 前端
├── scripts/            # 演示、评估、消融实验脚本
├── tests/              # 单元测试
├── weights/            # 模型权重（yolov8s.pt 等）
├── outputs/            # 处理结果输出
├── data/               # 数据集（UA-DETRAC 等）
└── requirements.txt
```

---

## 技术栈

**后端**

- Python 3.9+
- YOLOv8（Ultralytics）
- NumPy / SciPy（EKF 矩阵运算、匈牙利算法）
- OpenCV（视频处理、可视化）
- FastAPI + Uvicorn（API 服务）
- Pydantic（数据校验）

**前端**

- React 18 + TypeScript
- Vite
- Tailwind CSS
- Zustand（状态管理）
- lucide-react（图标）
- Axios（HTTP 请求）
