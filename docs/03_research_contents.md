# 第三章 研究内容

## 3.1 课题主要研究内容

本课题的核心研究内容是：**在 CPU 环境下，设计并实现一个基于扩展卡尔曼滤波（EKF）和 CTRV 运动模型的多目标检测、跟踪与轨迹预测系统，并配套 Web 可视化界面供演示和评估使用。**

具体包含以下几个子课题：

1. **目标检测子系统**：集成 YOLOv8n 检测器，支持 Ultralytics 和 ONNX 两种推理后端，输出带置信度的边界框。
2. **EKF 状态估计**：基于 CTRV 运动模型实现 7 维状态向量的 EKF，包含预测步、更新步和数值稳定性处理。
3. **三阶段数据关联**：设计融合 IoU、Mahalanobis 距离、中心距离的三阶段匈牙利匹配策略。
4. **轨迹生命周期管理**：实现 Tentative → Confirmed → Lost → Removed 四状态机。
5. **轨迹预测**：基于 EKF 状态递推，预测轨迹未来 1/5/10 帧的中心位置。
6. **Web 服务与前端**：提供 FastAPI 后端接口（视频上传处理、摄像头实时预测）和 React 前端界面。
7. **评估框架**：搭建检测、跟踪、预测三类指标的评估框架（脚本和指标模块）。

---

## 3.2 系统模块组成

```
EKFsystem/
├── src/ekf_mot/
│   ├── core/           ← 类型定义、接口、常量、配置加载
│   ├── data/           ← 视频读取、帧采样、数据格式转换
│   ├── detection/      ← YOLOv8 检测器（ultralytics / ONNX 两后端）
│   ├── filtering/      ← EKF 核心（ekf.py）、CTRV 模型、雅可比矩阵、噪声矩阵
│   ├── tracking/       ← 多目标跟踪器、数据关联、轨迹对象、生命周期管理
│   ├── prediction/     ← 轨迹预测器、平滑器、不确定性估计
│   ├── metrics/        ← 检测/跟踪/预测/运行时四类评估指标
│   ├── serving/        ← FastAPI 服务接口（api.py、service.py、schemas.py）
│   ├── visualization/  ← 检测框、轨迹、预测轨迹、协方差椭圆绘制
│   └── utils/          ← 日志、计时、几何工具、随机种子
├── configs/            ← YAML 配置文件（分 base / detector / tracker / exp 层级）
├── scripts/            ← 可执行脚本（demo、tracking、evaluate、webcam 等）
├── frontend/           ← React + TypeScript + Vite 前端（三页面）
└── tests/              ← pytest 单元测试（EKF、CTRV、关联、跟踪器）
```

---

## 3.3 各模块职责说明

### 3.3.1 检测模块（`detection/`）

| 文件 | 职责 |
|------|------|
| `yolo_ultralytics.py` | 调用 ultralytics YOLOv8 推理，返回标准化 Detection 列表 |
| `yolo_onnx.py` | 调用 ONNX Runtime 推理，适合部署场景 |
| `postprocess.py` | NMS、置信度过滤、坐标格式转换 |
| `evaluator.py` | 计算检测 Precision/Recall/F1（相对标注数据） |

### 3.3.2 EKF 滤波模块（`filtering/`）

| 文件 | 职责 |
|------|------|
| `ekf.py` | EKF 主类，预测步 + 更新步 + `predict_n_steps()` |
| `models/ctrv.py` | CTRV 非线性转移函数 `ctrv_predict()` 和雅可比 `ctrv_jacobian()` |
| `jacobians.py` | 观测矩阵 H（4×7 常数矩阵，提取 cx/cy/w/h） |
| `noise.py` | 过程噪声 Q、观测噪声 R、初始协方差 P 的构造函数 |
| `gating.py` | Mahalanobis 距离门控 |

### 3.3.3 跟踪模块（`tracking/`）

| 文件 | 职责 |
|------|------|
| `multi_object_tracker.py` | 跟踪器主入口，逐帧驱动关联、更新、生命周期管理 |
| `association.py` | 三阶段匈牙利匹配（Stage A/B/C） |
| `cost.py` | IoU 代价矩阵、融合代价矩阵 |
| `track.py` | 单条轨迹对象（包含 EKF 实例） |
| `track_manager.py` | 轨迹集合增删管理 |
| `lifecycle.py` | 轨迹状态转换逻辑（n_init / max_age） |
| `track_state.py` | TrackState 枚举（Tentative/Confirmed/Lost/Removed） |

### 3.3.4 轨迹预测模块（`prediction/`）

| 文件 | 职责 |
|------|------|
| `trajectory_predictor.py` | 批量预测满足质量门限的轨迹未来位置（中心点） |
| `smoother.py` | Fixed-lag 轨迹平滑（可选，离线模式） |
| `uncertainty.py` | 预测置信度估算 |

### 3.3.5 Web 服务（`serving/`）

| 文件 | 职责 |
|------|------|
| `api.py` | FastAPI 路由，包含 `/health`、`/predict/frame`、`/predict/video/start`、`/predict/video/status/{task_id}`、`/outputs/{filename}`、`/reset` |
| `service.py` | 业务逻辑层，处理单帧和视频处理任务，调用跟踪器 |
| `schemas.py` | Pydantic 请求/响应模型 |

### 3.3.6 前端（`frontend/`）

| 页面 / 组件 | 功能 |
|---|---|
| `DashboardPage.tsx` | 系统总控台，后端连接状态、功能入口卡片、技术栈说明 |
| `UploadPredictPage.tsx` | 视频上传预测，拖拽上传 → 后台异步处理 → 结果视频播放 + 下载 |
| `CameraPredictPage.tsx` | 摄像头实时预测，逐帧调用 `/predict/frame`，Canvas 绘制目标框和预测轨迹 |

---

## 3.4 系统重点与难点

### 3.4.1 重点

1. **EKF 与 CTRV 模型的正确实现**：雅可比矩阵推导、数值稳定性（`omega ≈ 0` 退化、Joseph 更新公式、协方差正定性保证）。
2. **三阶段关联策略**：按轨迹状态（Confirmed/Tentative）和检测置信度（高/低）分层处理，减少遮挡和漏检导致的 ID 切换。
3. **异步视频处理架构**：FastAPI 后台线程处理长视频，前端轮询进度，避免 HTTP 超时。

### 3.4.2 难点

1. **运动模型退化处理**：CTRV 在 `omega → 0` 时分母趋于零，需要用 CV 分支平滑过渡（`ctrv.py:41~48`）。
2. **视频编码兼容性**：OpenCV 默认 `mp4v` 编码输出的 FMP4 格式浏览器无法播放，需改用 `avc1` 通过 Windows Media Foundation 编码 H.264（`service.py:142~149`）。
3. **坐标一致性**：检测框坐标（像素）、EKF 状态（像素）、前端 Canvas 渲染坐标需要在不同模块间保持一致，尤其是 `anchor_mode`（几何中心 vs 底部中心）的统一处理。
4. **Bootstrap 初始化**：EKF 状态中速度、航向角在轨迹初始帧未知，用零值初始化会导致前几帧预测偏差大，通过 `set_kinematics()` 接口在轨迹确认前根据前几帧差分估计运动状态。
