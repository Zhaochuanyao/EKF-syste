# 第四章 毕业设计论文章节结构

> 本文档严格对照课题任务书的章节要求，说明每章应写什么内容及对应的代码/文档位置。

---

## 课题定义（来自任务书）

> 本课题实现从视频帧中目标检测到目标轨迹预测的一体化系统。该研究对于提高智能监控系统的实时性与稳定性、辅助无人车路径规划、以及多目标跟踪场景下的运动建模均具有较高的实际应用价值。

---

## 论文章节结构（与任务书对应）

---

### 第一章 绪论

#### 1.1 研究背景与意义

**任务书要求**：介绍目标检测和轨迹预测在现代技术中的重要性，如自动驾驶、安防监控等领域；说明为什么需要准确地识别目标并预测其未来的运动趋势。

**写作方向**：
- 自动驾驶：感知层需要实时感知周围动态目标并预测其短期轨迹，供路径规划使用
- 安防监控：传统人工查看录像效率低，智能检测+跟踪系统可自动实现异常行为预警
- 无人车路径规划：实时障碍物轨迹预测减少碰撞风险
- 本系统价值：全流程、CPU 可用、有 Web 界面，具备演示价值

**对应文档**：[01_background_and_significance.md](01_background_and_significance.md)

---

#### 1.2 国内外研究现状

**任务书要求**：简要回顾现有目标检测和轨迹预测方法，包括常用的图像处理技术和滤波方法；指出扩展卡尔曼滤波在处理非线性系统中的优势。

**写作方向**：
- 目标检测：Faster R-CNN（精度高但慢）、SSD（速度较快）、YOLO 系列（实时）→ 选用 YOLOv8n
- 目标跟踪：线性 KF + CV（DeepSORT前身）、EKF + CTRV（本项目）、SORT、DeepSORT
- EKF 优势：通过一阶泰勒展开处理非线性运动方程（CTRV），比 CV 模型对曲线运动更准确

**对应文档**：[02_related_work.md](02_related_work.md)

---

#### 1.3 主要研究内容

**任务书要求**：概述本设计要完成的核心任务，即构建一个结合目标检测和扩展卡尔曼滤波的完整系统。

**写作方向**：
- 基于 YOLOv8n 的目标检测子系统
- 基于 CTRV 的 EKF 状态估计与滤波
- 三阶段匈牙利数据关联
- 轨迹生命周期管理（四状态机）
- 多步递推轨迹预测（未来 1/5/10 帧）
- Web 服务与前端界面

**对应文档**：[03_research_contents.md](03_research_contents.md)

---

#### 1.4 论文组织结构

**写作方向**：说明各章内容安排，即本文档自身的结构描述。

---

### 第二章 目标检测模块设计与实现

#### 2.1 目标检测算法选择

**任务书要求**：确定用于识别图像或视频中目标的具体方法；说明选择该方法的原因（实时性、准确率等）。

**写作方向**：
- 对比 Faster R-CNN / SSD / YOLOv8n 的速度与精度权衡
- YOLOv8n 在 CPU 上可达 8~20 FPS（640×640 输入），适合毕业设计硬件
- 支持 ultralytics 和 ONNX 两种推理后端

**对应代码**：`src/ekf_mot/detection/yolo_ultralytics.py`，`yolo_onnx.py`

---

#### 2.2 数据准备与预处理

**任务书要求**：描述用于训练和测试检测模型的数据集，以及对原始数据进行的图像处理步骤。

**写作方向**：
- 本项目使用预训练 YOLOv8n（COCO 数据集训练），无需自行训练
- 预处理：视频帧抽取（frame_skip 跳帧）、BGR→RGB 颜色空间转换、resize 到 640×640
- 后处理：NMS、置信度过滤、坐标格式转换（xyxy → cx,cy,w,h）

**对应代码**：`src/ekf_mot/data/video_reader.py`，`src/ekf_mot/detection/postprocess.py`

---

#### 2.3 模型训练与优化

**任务书要求**：阐述目标检测模型的训练过程，包括如何调整参数以提高检测的准确率和速度。

**写作方向**：
- 本项目使用 YOLOv8n 官方预训练权重（无需自行训练）
- 性能调优参数：`conf`（置信度阈值）、`iou`（NMS 阈值）、`imgsz`（输入分辨率）、`max_det`（最大检测数量）
- 跳帧参数 `frame_skip=2` 可在牺牲少量精度的同时提升处理速度

**对应配置**：`configs/exp/demo_cpu.yaml`，`configs/detector/yolov8n.yaml`

---

#### 2.4 检测结果输出

**任务书要求**：明确目标检测模块输出的数据格式，例如目标的边界框（位置信息）。

**写作方向**：
- 输出标准 `Detection` 对象列表（`src/ekf_mot/core/types.py`）
- 每个 Detection 包含：`bbox=[x1,y1,x2,y2]`、`score`（置信度）、`class_id`（类别编号）、`class_name`（类别名称）
- 传入 EKF 时转换为观测向量 `z = [cx, cy, w, h]`

---

### 第三章 扩展卡尔曼滤波（EKF）原理与建模

#### 3.1 滤波方法基础

**任务书要求**：简要介绍卡尔曼滤波的基本思想，及其在处理线性系统中的应用。

**写作方向**：KF 的两步流程（预测步、更新步），贝叶斯估计解释，线性高斯假设

---

#### 3.2 扩展卡尔曼滤波理论

**任务书要求**：解释为什么需要使用"扩展"版本（即处理非线性问题），以及其核心思想（通过线性化近似非线性）。

**写作方向**：CTRV 转移方程非线性 → 雅可比矩阵一阶泰勒展开线性化

**对应文档**：[05_ekf_modeling.md §5.4](05_ekf_modeling.md)

---

#### 3.3 系统状态建模

**任务书要求**：定义目标运动的状态向量（如位置、速度等）和观测向量（如检测模块输出的位置）。

**写作方向**：`x = [cx, cy, v, theta, omega, w, h]^T`（7维），`z = [cx, cy, w, h]^T`（4维）

**对应文档**：[05_ekf_modeling.md §5.1](05_ekf_modeling.md)

---

#### 3.4 运动与观测模型建立

**任务书要求**：建立描述目标运动规律（状态转移）和检测测量误差（观测）的数学模型。

**写作方向**：CTRV 非线性转移方程（含 omega≈0 退化）、线性观测方程 z=Hx

**对应文档**：[05_ekf_modeling.md §5.2, §5.3](05_ekf_modeling.md)

---

#### 3.5 噪声特性分析

**任务书要求**：分析系统和观测过程中可能存在的随机误差（噪声）的特性。

**写作方向**：过程噪声 Q（运动模型不确定性）、观测噪声 R（检测器精度误差）、置信度自适应 R

**对应文档**：[05_ekf_modeling.md §5.5](05_ekf_modeling.md)，`configs/tracker/ekf_ctrv.yaml`

---

### 第四章 目标跟踪与轨迹预测系统设计与实现

#### 4.1 数据关联

**任务书要求**：解决多目标场景中，如何将当前检测到的目标与历史轨迹中的目标进行正确匹配的问题。

**写作方向**：三阶段匈牙利关联（Stage A/B/C），融合 IoU + Mahalanobis + 中心距离代价

**对应文档**：[06_tracking_and_prediction.md §6.1](06_tracking_and_prediction.md)，`src/ekf_mot/tracking/association.py`

---

#### 4.2 扩展卡尔曼滤波的融合应用

**任务书要求**：详细描述如何将检测模块输出的观测数据输入到 EKF 中，进行预测和校正。

**写作方向**：逐帧驱动流程（预测步→关联→更新步），`update()` 方法将观测值融合到状态估计中

**对应文档**：[06_tracking_and_prediction.md §6.2](06_tracking_and_prediction.md)

---

#### 4.3 目标轨迹平滑与更新

**任务书要求**：阐述滤波器如何根据最新的观测值修正目标的状态，形成平滑的运动轨迹。

**写作方向**：EKF 更新步（Joseph 形式协方差更新），历史中心点保存，可选 Fixed-lag 平滑

**对应文档**：[06_tracking_and_prediction.md §6.4.1](06_tracking_and_prediction.md)

---

#### 4.4 运动轨迹预测

**任务书要求**：描述利用滤波器的预测功能，在没有最新观测值的情况下，推算目标在未来一段时间内的可能位置。

**写作方向**：`predict_n_steps(n, dt)` 不修改内部状态的递推预测，预测质量门限，预测置信度

**对应文档**：[06_tracking_and_prediction.md §6.4.2](06_tracking_and_prediction.md)

---

#### 4.5 系统整体框架搭建

**任务书要求**：展示目标检测、数据关联和 EKF 三个模块如何协同工作，形成完整的系统流程。

**写作方向**：数据流图（见 README.md 架构图），模块协同说明

**对应文档**：[03_research_contents.md §3.2](03_research_contents.md)

---

### 第五章 实验与结果分析

#### 5.1 实验环境搭建

**对应文档**：[07_experiment_setup.md](07_experiment_setup.md)（可直接引用）

---

#### 5.2 数据集与评估指标

**任务书要求**：确定用于测试系统性能的数据集，并选择合适的指标（准确率、跟踪精度、预测误差等）进行量化评估。

**写作方向**：
- 视频数据：`assets/samples/demo.mp4`（演示视频），后续可接入 MOT17 / UA-DETRAC
- 检测指标：Precision、Recall、F1（`src/ekf_mot/metrics/detection_metrics.py`）
- 跟踪指标：MOTA、MOTP、ID Switch（`src/ekf_mot/metrics/tracking_metrics.py`）
- 预测指标：ADE（平均位移误差）、FDE（最终位移误差）（`src/ekf_mot/metrics/prediction_metrics.py`）
- 运行时：FPS、平均帧耗时（`src/ekf_mot/metrics/runtime_metrics.py`）

---

#### 5.3 系统性能测试

**任务书要求**：针对不同的运动场景（匀速、变速、转弯等）进行测试。

**写作方向**：
- 场景 1：行人（匀速直走）→ `configs/exp/demo_person_accuracy.yaml`
- 场景 2：车辆（含转弯）→ `configs/exp/demo_vehicle_accuracy.yaml`
- 场景 3：混合场景（行人 + 车辆）→ `configs/exp/demo_cpu.yaml`

---

#### 5.4 结果对比与分析

**任务书要求**：将基于 EKF 的轨迹与仅使用检测结果的轨迹进行对比，突出滤波器的优势；分析系统的不足之处和误差来源。

**写作方向**：
- EKF 轨迹 vs 原始检测框轨迹（抖动对比，视频截图）
- EKF 遮挡恢复能力（Lost 状态持续预测）
- 不足：无 ReID，ID 切换在严重遮挡时仍存在；CPU 限制导致帧率较低
- 误差来源：检测精度、CTRV 模型假设偏差、噪声参数未自适应调整

---

### 参考文献

可引用的核心文献方向（毕业设计不要求完整引文格式，列出即可）：
- Bewley et al., "Simple Online and Realtime Tracking" (SORT, 2016)
- Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (DeepSORT, 2017)
- Jocher et al., "Ultralytics YOLOv8" (2023)
- Thrun et al., "Probabilistic Robotics" (EKF/CTRV 理论来源)
