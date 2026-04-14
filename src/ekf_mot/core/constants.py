"""
常量定义模块 - 集中管理系统中使用的所有常量
"""

# ── 轨迹状态名称 ──────────────────────────────────────────────
TRACK_STATE_TENTATIVE = "Tentative"
TRACK_STATE_CONFIRMED = "Confirmed"
TRACK_STATE_LOST = "Lost"
TRACK_STATE_REMOVED = "Removed"

# ── EKF 状态向量索引 ──────────────────────────────────────────
# 状态向量: [cx, cy, v, theta, omega, w, h]
IDX_CX = 0      # 中心x坐标
IDX_CY = 1      # 中心y坐标
IDX_V = 2       # 速度模长
IDX_THETA = 3   # 航向角（弧度）
IDX_OMEGA = 4   # 角速度（弧度/秒）
IDX_W = 5       # 目标框宽度
IDX_H = 6       # 目标框高度

STATE_DIM = 7   # 状态向量维度

# ── 观测向量索引 ──────────────────────────────────────────────
# 观测向量: [cx, cy, w, h]
MEAS_IDX_CX = 0
MEAS_IDX_CY = 1
MEAS_IDX_W = 2
MEAS_IDX_H = 3

MEAS_DIM = 4    # 观测向量维度

# ── 检测器后端 ────────────────────────────────────────────────
BACKEND_ULTRALYTICS = "ultralytics"
BACKEND_ONNX = "onnx"

# ── 默认阈值 ──────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_GATING_THRESHOLD = 9.4877   # chi2(0.95, df=4)
DEFAULT_OMEGA_THRESHOLD = 0.001     # omega接近0的判断阈值

# ── 轨迹生命周期默认值 ────────────────────────────────────────
DEFAULT_N_INIT = 3
DEFAULT_MAX_AGE = 20

# ── 可视化颜色（BGR格式）────────────────────────────────────────
COLOR_TENTATIVE = (128, 128, 128)   # 灰色
COLOR_CONFIRMED = (0, 255, 0)       # 绿色
COLOR_LOST = (0, 165, 255)          # 橙色
COLOR_FUTURE = (0, 255, 255)        # 黄色
COLOR_COVARIANCE = (255, 0, 255)    # 紫色
COLOR_TEXT = (255, 255, 255)        # 白色

# ── COCO 类别名称（部分）────────────────────────────────────────
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
}
