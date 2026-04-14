# 第五章 EKF 建模说明

> 本文档严格对应代码实现，所有公式均可在源码中找到对应逻辑。
> 核心文件：`src/ekf_mot/filtering/ekf.py`、`src/ekf_mot/filtering/models/ctrv.py`、`src/ekf_mot/filtering/noise.py`、`src/ekf_mot/filtering/jacobians.py`

---

## 5.1 状态向量与观测向量

### 5.1.1 状态向量（7 维）

```
x = [cx, cy, v, theta, omega, w, h]^T
```

| 分量 | 含义 | 单位 | 代码索引常量 |
|------|------|------|------------|
| `cx` | 目标框中心 x 坐标 | 像素 | `IDX_CX = 0` |
| `cy` | 目标框中心 y 坐标 | 像素 | `IDX_CY = 1` |
| `v` | 速度模长 | 像素/秒 | `IDX_V = 2` |
| `theta` | 航向角（x 轴正方向为 0，逆时针为正） | 弧度 | `IDX_THETA = 3` |
| `omega` | 角速度（转率） | 弧度/秒 | `IDX_OMEGA = 4` |
| `w` | 目标框宽度 | 像素 | `IDX_W = 5` |
| `h` | 目标框高度 | 像素 | `IDX_H = 6` |

> 代码位置：`src/ekf_mot/core/constants.py`，`src/ekf_mot/filtering/ekf.py:28~38`

**设计说明**：将目标尺寸 `(w, h)` 纳入状态向量，使得 EKF 能够对目标的大小变化进行平滑估计，避免检测框尺寸抖动直接传递到输出。

### 5.1.2 观测向量（4 维）

```
z = [cx, cy, w, h]^T
```

观测向量直接来自目标检测器的输出（YOLOv8n 边界框转换为中心点格式），不包含速度和角度信息——这些隐含量由 EKF 从序列观测中自动估计。

---

## 5.2 CTRV 运动模型

CTRV（Constant Turn Rate and Velocity，恒转率恒速度）模型假设目标在一个时间步内保持速度模长和转率不变。

### 5.2.1 非线性状态转移方程 f(x, dt)

当 `|omega| >= omega_threshold`（非直线运动）：

```
cx_{t+1} = cx_t + (v/omega) * (sin(theta + omega*dt) - sin(theta))
cy_{t+1} = cy_t + (v/omega) * (-cos(theta + omega*dt) + cos(theta))
theta_{t+1} = theta_t + omega * dt
v_{t+1}   = v_t        （速度模长保持不变）
omega_{t+1} = omega_t  （角速度保持不变）
w_{t+1}   = w_t        （尺寸保持不变）
h_{t+1}   = h_t
```

当 `|omega| < omega_threshold`（近似直线运动，CV 退化）：

```
cx_{t+1} = cx_t + v * cos(theta) * dt
cy_{t+1} = cy_t + v * sin(theta) * dt
theta_{t+1} = theta_t  （航向不变）
```

> 代码位置：`src/ekf_mot/filtering/models/ctrv.py:15~64`，退化判断在第 41 行

**为什么要处理 omega ≈ 0 的情况**：当 `omega → 0` 时，标准 CTRV 公式的分母 `omega` 趋于零，会导致数值溢出。通过设置 `omega_threshold = 0.001`（可配置，`configs/tracker/ekf_ctrv.yaml`），当角速度极小时切换到 CV 退化分支，保证数值稳定。

### 5.2.2 为什么选用 CTRV 而非 CV

| 比较项 | CV 模型 | CTRV 模型（本项目） |
|--------|---------|-------------------|
| 状态维度 | 6（cx,cy,vx,vy,w,h） | 7（cx,cy,v,theta,omega,w,h） |
| 转弯建模 | ❌ 不支持 | ✅ 通过 omega 建模 |
| 直线运动 | ✅ | ✅（omega=0 自动退化为 CV） |
| 适用场景 | 行人直走 | 车辆转弯、行人变向 |
| 滤波器类型 | 线性 KF | EKF（非线性需要线性化） |

---

## 5.3 观测模型

观测方程为线性映射：

```
z = H * x
```

其中 H 为 4×7 观测矩阵（常数矩阵）：

```
H = [[1, 0, 0, 0, 0, 0, 0],   # cx
     [0, 1, 0, 0, 0, 0, 0],   # cy
     [0, 0, 0, 0, 0, 1, 0],   # w
     [0, 0, 0, 0, 0, 0, 1]]   # h
```

> 代码位置：`src/ekf_mot/filtering/jacobians.py`，常量 `H_MATRIX`

因为观测方程是线性的，观测的雅可比矩阵 $\partial h / \partial x = H$ 就是 H 本身（不需要数值微分）。

---

## 5.4 Jacobian 线性化

EKF 用一阶泰勒展开将非线性状态转移函数 f 线性化，得到状态转移雅可比矩阵 F（7×7）：

```
F[i,j] = df_i / dx_j
```

**CV 退化情形**（`|omega| < omega_threshold`）：

```
F[cx, v]     = cos(theta) * dt
F[cx, theta] = -v * sin(theta) * dt
F[cy, v]     = sin(theta) * dt
F[cy, theta] = v * cos(theta) * dt
```

**标准 CTRV 情形**：

```
F[cx, v]     = (sin(theta+omega*dt) - sin(theta)) / omega
F[cx, theta] = (v/omega) * (cos(theta+omega*dt) - cos(theta))
F[cx, omega] = (v/omega)*cos(theta+omega*dt)*dt - (v/omega^2)*(sin(theta+omega*dt)-sin(theta))
F[cy, v]     = (-cos(theta+omega*dt) + cos(theta)) / omega
F[cy, theta] = (v/omega) * (sin(theta+omega*dt) - sin(theta))
F[cy, omega] = (v/omega)*sin(theta+omega*dt)*dt - (v/omega^2)*(-cos(theta+omega*dt)+cos(theta))
F[theta, omega] = dt
```

其余对角元素为 1，非对角元素为 0。

> 代码位置：`src/ekf_mot/filtering/models/ctrv.py:67~132`

---

## 5.5 过程噪声与观测噪声

### 5.5.1 过程噪声矩阵 Q

Q 矩阵反映运动模型本身的不确定性（即模型与真实运动的偏差）。本项目 Q 由三类噪声参数构造：

| 参数 | 含义 | 默认值 | 配置位置 |
|------|------|--------|---------|
| `std_acc` | 加速度噪声标准差 | 2.0 | `configs/tracker/ekf_ctrv.yaml` |
| `std_yaw_rate` | 角速度变化噪声 | 0.5 | 同上 |
| `std_size` | 尺寸变化噪声 | 0.1 | 同上 |

> 构造函数：`src/ekf_mot/filtering/noise.py:build_process_noise_Q()`

### 5.5.2 观测噪声矩阵 R

R 矩阵反映检测器测量误差。本项目支持**置信度自适应 R**（`score_adaptive=true`）：检测置信度越低，R 中位置分量越大，表示对该观测的信任度越低。

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `std_cx` | 中心 x 观测噪声 | 5.0 像素 |
| `std_cy` | 中心 y 观测噪声 | 5.0 像素 |
| `std_w` | 宽度观测噪声 | 10.0 像素 |
| `std_h` | 高度观测噪声 | 10.0 像素 |

> 构造函数：`src/ekf_mot/filtering/noise.py:build_measurement_noise_R()`

---

## 5.6 EKF 两步流程

### 5.6.1 预测步（Predict）

```
x_pred = f(x, dt)             # CTRV 非线性传播
F = ctrv_jacobian(x, dt)      # 计算雅可比
Q = build_process_noise_Q(dt) # 过程噪声
P_pred = F * P * F^T + Q      # 协方差传播
```

**数值稳定性保证**（`ekf.py:179~185`）：
- 对称化：`P = (P + P^T) / 2`
- 正定性：对角元素小于 1e-6 时裁剪为 1e-6

### 5.6.2 更新步（Update）

```
z_pred = H * x_pred           # 预测观测
y = z - z_pred                # 新息（残差）
S = H * P * H^T + R           # 新息协方差
K = P * H^T * S^{-1}          # 卡尔曼增益
x = x_pred + K * y            # 状态更新
P = (I - K*H) * P * (I-K*H)^T + K*R*K^T  # Joseph 形式更新（数值稳定）
```

**Joseph 形式**（`ekf.py:257~259`）：相比标准 `P = (I-KH)*P` 更新，Joseph 形式在数值上更稳定，即使 K 有小的计算误差也能保持 P 的正定性。

---

## 5.7 为什么使用 EKF 而非普通 KF

| 对比项 | 普通 KF | EKF（本项目） |
|--------|---------|-------------|
| 适用系统 | 线性 | 非线性（通过 Jacobian 线性化） |
| 运动模型 | 必须是线性（CV） | 可以是非线性（CTRV） |
| 实现复杂度 | 低 | 中等（需推导 Jacobian） |
| 曲线运动精度 | 差（线性近似误差大） | 好（CTRV 直接建模弧线） |
| CPU 开销 | 低 | 略高（矩阵运算）但仍可实时 |

本项目目标场景包含车辆转弯等非直线运动，因此 EKF + CTRV 是比线性 KF + CV 更准确的选择。

---

## 5.8 Bootstrap 运动状态初始化

EKF 初始化时，速度 `v`、航向 `theta`、角速度 `omega` 均设为 0（`ekf.py:116~120`），因为第一帧无法直接观测这些量。这会导致轨迹确认前几帧的预测位置偏差较大（滤波收敛慢）。

本项目通过 `set_kinematics()` 接口（`ekf.py:296~320`）在轨迹进入 Confirmed 状态之前，根据前几帧中心点差分估算速度和方向，将估算值写入 EKF 状态，加速收敛（bootstrap 初始化策略）。
