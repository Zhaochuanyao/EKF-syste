# 第六章 跟踪与预测系统实现

> 核心文件：`src/ekf_mot/tracking/`、`src/ekf_mot/prediction/`、`src/ekf_mot/serving/`、`src/ekf_mot/visualization/`

---

## 6.1 数据关联

数据关联解决的问题是：**给定当前帧的 N 个检测框，将它们与上一帧遗留的 M 条轨迹正确配对**。配对错误会导致目标 ID 切换，降低跟踪连续性。

### 6.1.1 三阶段关联策略

本系统采用三阶段匹配（`src/ekf_mot/tracking/association.py`），设计理由是：不同状态的轨迹、不同置信度的检测框，应使用差异化的匹配策略。

```
Stage A：(Confirmed + Lost) 轨迹  ×  高置信度检测框
         代价函数：IoU + Mahalanobis 距离 + 中心距离（三者加权融合）
         严格门控（Mahalanobis 距离 > 9.49 则不可匹配）

Stage B：Stage A 未匹配的轨迹   ×  低置信度检测框
         代价函数：纯 IoU（阈值放宽到 0.4）
         目的：找回弱观测目标（遮挡/远距离目标）

Stage C：Tentative 轨迹          ×  Stage A 未匹配的高置信度检测框
         代价函数：纯 IoU（阈值 0.3）
         目的：为新目标候选轨迹快速匹配观测
```

### 6.1.2 代价函数说明

**IoU 代价**（`src/ekf_mot/tracking/cost.py:iou_cost_matrix()`）：

```
cost_IoU = 1 - IoU(predicted_bbox, detected_bbox)
```

用 EKF 预测的位置（而非上一帧位置）生成 predicted_bbox，再与检测框计算 IoU，使得运动越快的目标也能正确匹配。

**Mahalanobis 距离门控**（`src/ekf_mot/filtering/gating.py`）：

```
d_mahal = (z - z_pred)^T * S^{-1} * (z - z_pred)
```

门控阈值 9.4877 对应 4 自由度卡方分布 95% 置信区间，超过阈值的匹配对直接置为 `inf`（不可匹配）。Mahalanobis 距离考虑了 EKF 预测的不确定性，距离越远且不确定性越小，则对匹配越严格。

**融合代价**（Stage A）：

```
cost = iou_weight * cost_IoU + mahal_weight * cost_Mahal + center_weight * cost_Center
```

默认权重：`iou_weight=0.4, mahal_weight=0.4, center_weight=0.2`，可通过配置文件调整（`configs/tracker/association.yaml`）。

### 6.1.3 匈牙利算法

代价矩阵构造完成后，使用 `scipy.optimize.linear_sum_assignment` 求解最优分配，时间复杂度 O(N³)，对每帧目标数量（通常 < 50）完全满足实时要求。代价超过阈值的匹配对被拒绝（视为未匹配）。

> 代码：`src/ekf_mot/tracking/association.py:hungarian_match()`

---

## 6.2 EKF 与检测结果的融合

每帧处理流程（`src/ekf_mot/tracking/multi_object_tracker.py`）：

```
1. 用检测器获取当前帧检测结果（Detection 列表）
2. 对所有活跃轨迹执行 EKF 预测步（predict()）
3. 三阶段数据关联（associate()）
4. 已匹配的轨迹：用对应检测框执行 EKF 更新步（update()）
5. 未匹配的轨迹：递增 time_since_update，超过 max_age 则移除
6. 未匹配的检测框：创建新的 Tentative 轨迹
7. 生命周期状态转换（Tentative → Confirmed / Lost → Removed）
```

**关键设计**：EKF 预测步在关联之前执行，这意味着关联时使用的是"预测位置"而非"历史位置"，更适合快速运动目标的匹配。

---

## 6.3 轨迹生命周期管理

轨迹经历四个状态（`src/ekf_mot/tracking/track_state.py`、`lifecycle.py`）：

```
新建 → Tentative（候选）
    ↓ 连续命中 >= n_init（默认3帧）
Confirmed（确认）
    ↓ 连续丢失帧数超过 max_age（默认15帧）  
Lost（丢失）
    ↓ 进一步超过额外阈值或 omega 过大
Removed（移除）
```

| 状态 | 参与关联 | 是否输出 | 说明 |
|------|---------|---------|------|
| Tentative | Stage C（纯 IoU） | 否 | 等待确认，避免误检生成轨迹 |
| Confirmed | Stage A（融合代价）| ✅ | 正常输出检测框、轨迹、预测 |
| Lost | Stage A | ✅（仅框） | 短暂遮挡恢复期 |
| Removed | 否 | 否 | 已从跟踪器中删除 |

---

## 6.4 轨迹平滑与预测

### 6.4.1 轨迹平滑

由于 EKF 在每次更新后状态已经是滤波后的平滑估计，历史轨迹路径（用于可视化的轨迹线）通过保存每帧 `confirmed` 状态下的 `(cx, cy)` 中心点来绘制，`track_history_len=20` 控制轨迹线保留帧数（`configs/exp/demo_cpu.yaml`）。

可选地，`src/ekf_mot/prediction/smoother.py` 实现了 Fixed-lag 平滑，在离线模式下对历史轨迹进行回顾平滑，进一步降低轨迹抖动。

### 6.4.2 未来轨迹预测

预测模块（`src/ekf_mot/prediction/trajectory_predictor.py`）对满足质量门限的 Confirmed 轨迹执行**多步递推预测**：

**质量门限**（`is_eligible()`）：
- 状态为 Confirmed
- 命中次数 `hits >= min_hits_for_prediction`（默认 3）
- 当前帧有命中（`time_since_update == 0`）
- 位置协方差迹 < `max_position_cov_trace`（避免不确定性过高时的外推失真）

**预测方法**（`predict_track()`）：
1. 调用 `ekf.predict_n_steps(max_steps, dt)` 递推计算未来状态
2. `predict_n_steps` 在临时副本上执行，不修改 EKF 真实状态
3. 从预测状态中提取 `(cx, cy)`，按 `future_steps=[1, 5, 10]` 返回

**预测置信度**（`compute_prediction_confidence()`）：
综合位置协方差不确定性、命中次数、运动状态有效性、时效性四个因子，输出 `[0, 1]` 的置信度分数，供前端可视化用于决定是否显示预测点。

---

## 6.5 输出格式

### 6.5.1 视频输出

使用 `src/ekf_mot/visualization/` 下各模块在视频帧上绘制：
- 检测框（带 ID 标签、类别、置信度）
- 轨迹历史线（最近 N 帧中心点连线）
- 预测轨迹（未来 1/5/10 帧中心点 + 连线，青色虚线）
- 协方差椭圆（可选，反映位置不确定性）

输出视频编解码：`avc1`（H.264），由 Windows Media Foundation 编码，浏览器可直接播放（`src/ekf_mot/serving/service.py:142~149`）。

文件命名含时间戳，格式：`output_YYYYMMDD_HHMMSS.mp4`，保存于 `outputs/` 目录。

### 6.5.2 JSON 输出

每帧保存跟踪结果到 JSON 文件，格式示例：

```json
{
  "frame_id": 42,
  "tracks": [
    {
      "track_id": 3,
      "bbox": {"x1": 100, "y1": 200, "x2": 180, "y2": 320},
      "class_name": "person",
      "score": 0.87,
      "state": "confirmed",
      "center": [140, 260],
      "future_points": {
        "1": [142, 258],
        "5": [150, 250],
        "10": [162, 240]
      }
    }
  ]
}
```

### 6.5.3 CSV 输出

轨迹汇总数据保存为 CSV，包含：`frame_id, track_id, class_id, x1, y1, x2, y2, score, state`，适合后续用 pandas 做统计分析和评估。

---

## 6.6 Web API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/predict/frame` | 单帧预测（摄像头实时模式），Base64 图像输入 |
| POST | `/predict/video/start` | 上传视频并启动异步处理任务，返回 `task_id` |
| GET | `/predict/video/status/{task_id}` | 轮询任务进度，返回 status/progress/result |
| GET | `/outputs/{filename}` | 获取处理完成的输出视频文件 |
| POST | `/reset` | 重置跟踪器状态（用于摄像头模式重新开始） |

异步架构说明：视频处理在后台线程中执行，前端每 2 秒轮询状态接口，避免 HTTP 超时（长视频处理可能需要数分钟）。

---

## 6.7 前端与后端联调

前端（`frontend/`）通过 `VITE_API_BASE_URL=http://localhost:8000` 连接后端。

**视频上传流程**：
```
用户拖拽视频文件
    → POST /predict/video/start（携带 FormData）
    → 获得 task_id
    → 每2秒 GET /predict/video/status/{task_id}
    → status == "done"
    → GET /outputs/{output_file} 渲染视频
```

**摄像头实时流程**：
```
浏览器 getUserMedia 获取摄像头流
    → 每帧(1000ms/fps) Canvas.toDataURL 截帧
    → POST /predict/frame（Base64 图像）
    → 返回 tracks 列表
    → Canvas 上绘制检测框和预测轨迹
```
