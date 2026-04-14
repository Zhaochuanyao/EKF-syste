# 第九章 数据集准备与格式转换

> 本文档说明 UA-DETRAC 和 MOT17 数据集的下载方法、转换流程，以及当前项目对数据集的支持程度。

---

## 9.1 当前项目对数据集的支持程度

| 功能 | 状态 | 说明 |
|------|------|------|
| 读取本地视频文件 | ✅ 完全支持 | `assets/samples/demo.mp4`，或任意 `.mp4/.avi/.mov` |
| UA-DETRAC XML → 内部 JSON | ✅ 完全支持 | `scripts/convert_annotations.py --format ua_detrac` |
| MOT17 gt.txt → 内部 JSON | ✅ 完全支持 | `scripts/convert_annotations.py --format mot17` |
| 内部 JSON GT 用于检测评估 | ✅ 完全支持 | `scripts/validate_detector.py --gt-json` |
| 纯视频检测统计（无 GT） | ✅ 完全支持 | `scripts/validate_detector.py --video` |
| 训练检测器 | ⚠️ 部分支持 | 框架已有 `detection/trainer.py`，需自备标注数据 |
| train/val/test 数据集切分 | ⚠️ 部分支持 | 现阶段通过文件夹命名手动区分，无自动切分工具 |
| HOTA / TrackEval 标准评估 | 🔲 待实现 | 下一轮补齐 |

---

## 9.2 内部统一数据格式

项目使用统一的 JSON 格式作为标注的内部表示，所有格式转换的输出均为此格式：

```json
{
  "dataset": "ua_detrac",
  "sequence": "MVI_20011",
  "fps": 25,
  "total_frames": 664,
  "frames": [
    {
      "frame_id": 0,
      "annotations": [
        {
          "id": 1,
          "bbox": [592.75, 378.14, 681.61, 441.99],
          "class_id": 0,
          "class_name": "car"
        }
      ]
    }
  ]
}
```

**字段说明**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `dataset` | string | 来源数据集名称 |
| `sequence` | string | 序列名称（对应视频文件名） |
| `fps` | int | 视频帧率 |
| `frames[*].frame_id` | int | 帧编号（从 0 或 1 开始，与原格式一致） |
| `frames[*].annotations[*].id` | int | 目标 ID（来自原始标注） |
| `frames[*].annotations[*].bbox` | [x1,y1,x2,y2] | 像素坐标，左上+右下角 |
| `frames[*].annotations[*].class_id` | int | 类别编号 |
| `frames[*].annotations[*].class_name` | string | 类别名称 |

---

## 9.3 UA-DETRAC 数据集

### 9.3.1 数据集简介

UA-DETRAC（University at Albany Detection and Tracking）是一个车辆检测与跟踪基准数据集：
- 100 个视频序列，共约 140,000 帧
- 在中国北京天安门广场和天安门隧道拍摄
- 目标类别：car、bus、van、others（均为车辆）
- 标注格式：XML，逐帧提供目标边界框

### 9.3.2 下载方法

**官方网站**：https://detrac-db.rit.albany.edu/

需要注册账号后下载：

| 文件 | 说明 | 大小 |
|------|------|------|
| `DETRAC-train-data.zip` | 训练集视频帧（图像序列） | ~1.5 GB |
| `DETRAC-test-data.zip` | 测试集视频帧 | ~0.5 GB |
| `DETRAC-Train-Annotations-XML.zip` | 训练集 XML 标注 | ~10 MB |

**备用下载（部分镜像）**：
```bash
# 如果官网无法直接下载，可尝试以下方式（不保证持续有效）
# 方法1：在 Kaggle 上搜索 "UA-DETRAC"
# 方法2：联系数据集作者获取下载链接
```

### 9.3.3 解压后目录结构

```
data/UA-DETRAC/
├── DETRAC-train-data/
│   ├── MVI_20011/           # 每个序列一个目录
│   │   ├── img00001.jpg
│   │   ├── img00002.jpg
│   │   └── ...
│   └── MVI_20012/
│       └── ...
├── DETRAC-test-data/
│   └── ...
└── DETRAC-Train-Annotations-XML/
    ├── MVI_20011.xml        # 每个序列一个 XML 文件
    ├── MVI_20012.xml
    └── ...
```

### 9.3.4 转换到内部格式

```bash
# 转换单个序列 XML
python scripts/convert_annotations.py \
    --format ua_detrac \
    --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/MVI_20011.xml \
    --output data/processed/ua_detrac/ \
    --validate

# 批量转换整个标注目录
python scripts/convert_annotations.py \
    --format ua_detrac \
    --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/ \
    --output data/processed/ua_detrac/ \
    --validate
```

### 9.3.5 UA-DETRAC 标注 XML 格式

```xml
<sequence name="MVI_20011">
  <frame density="7" num="0">
    <target_list>
      <target id="1">
        <box left="592.75" top="378.14" width="88.86" height="63.85" />
        <attribute orientation="rearview" speed="14.47"
                   vehicle_type="car" truncation_ratio="0" />
      </target>
    </target_list>
  </frame>
</sequence>
```

**字段说明**：
- `frame/@num`：帧编号（从 0 开始）
- `box/@left, @top`：左上角坐标（像素）
- `box/@width, @height`：宽高（像素）
- `attribute/@vehicle_type`：车辆类型（car/bus/van/others）

---

## 9.4 MOT17 数据集

### 9.4.1 数据集简介

MOT17（Multiple Object Tracking Benchmark 2017）是最常用的行人跟踪基准：
- 14 个序列（train 7 个，test 7 个）
- 每个序列提供 3 种检测器版本（DPM/FRCNN/SDP），共 21 个训练变体
- 目标类别以行人为主，另有少量骑车人等
- 标注格式：txt，MOTC（MOTChallenge）格式

### 9.4.2 下载方法

**官方网站**：https://motchallenge.net/data/MOT17/

```bash
# 直接下载（需要注册或直链）
# 文件：MOT17.zip（约 5.5 GB）
# 解压后放到 data/MOT17/
```

### 9.4.3 解压后目录结构

```
data/MOT17/
├── train/
│   ├── MOT17-02-FRCNN/      # 序列-检测器
│   │   ├── gt/
│   │   │   └── gt.txt       # Ground truth 标注
│   │   ├── img1/            # 视频帧图像
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   ├── det/
│   │   │   └── det.txt      # 检测器结果（可用作 baseline）
│   │   └── seqinfo.ini      # 序列元信息（含帧率、分辨率等）
│   ├── MOT17-04-FRCNN/
│   └── ...（共 7×3=21 个目录）
└── test/
    └── ...（无 gt.txt）
```

### 9.4.4 gt.txt 格式说明

```
frame_id, track_id, x, y, w, h, conf, class, visibility
```

| 列 | 含义 |
|----|------|
| frame_id | 帧编号（从 1 开始） |
| track_id | 目标 ID |
| x, y | 左上角坐标（像素） |
| w, h | 宽高（像素） |
| conf | 标注置信度（0=忽略区域，转换时会跳过） |
| class | 类别（1=行人，其他参见 MOTChallenge 说明） |
| visibility | 可见度 [0,1] |

### 9.4.5 转换到内部格式

```bash
# 转换单个序列
python scripts/convert_annotations.py \
    --format mot17 \
    --input data/MOT17/train/MOT17-02-FRCNN \
    --output data/processed/mot17/

# 批量转换整个 train/ 目录（21 个序列）
python scripts/convert_annotations.py \
    --format mot17 \
    --input data/MOT17/train/ \
    --output data/processed/mot17/ \
    --validate
```

---

## 9.5 运行检测评估

### 9.5.1 对视频文件运行检测（无 GT，只统计）

```bash
python scripts/validate_detector.py \
    --video assets/samples/demo.mp4 \
    --output outputs/detection/
```

输出（`outputs/detection/detection_metrics.json`）：
```json
{
  "mode": "detection_only",
  "total_frames_processed": 250,
  "total_detections": 1823,
  "avg_detections_per_frame": 7.29,
  "avg_inference_ms": 87.3,
  "avg_fps": 11.5
}
```

### 9.5.2 对照 GT 标注评估检测精度

需要先完成标注转换（§9.3.4 或 §9.4.5），并有对应的视频文件：

```bash
# UA-DETRAC 示例（需要 GT JSON 和对应视频）
python scripts/validate_detector.py \
    --video data/UA-DETRAC/DETRAC-train-data/MVI_20011/  \
    --gt-json data/processed/ua_detrac/MVI_20011.json \
    --output outputs/detection/ \
    --iou-threshold 0.5

# 输出指标示例
# Precision=0.7823  Recall=0.6541  F1=0.7124  AP50=0.6890
```

### 9.5.3 输出文件说明

```
outputs/detection/
└── detection_metrics.json   # 评估报告
```

报告结构：
```json
{
  "iou_threshold": 0.5,
  "num_frames": 664,
  "global": {
    "precision": 0.7823,
    "recall": 0.6541,
    "f1": 0.7124,
    "ap50": 0.6890,
    "tp": 4821,
    "fp": 1342,
    "fn": 2543,
    "num_frames": 664
  },
  "per_class": {
    "0": {
      "class_name": "car",
      "precision": 0.8012,
      ...
    }
  }
}
```

---

## 9.6 当前不支持的功能（诚实说明）

| 功能 | 原因 | 影响 |
|------|------|------|
| 从图像序列（非视频）读取帧 | `validate_detector.py` 仅支持 OpenCV VideoCapture | 如有图像序列数据集，需要先转为视频或扩展读取代码 |
| 自动 train/val/test 划分 | 当前无切分工具 | 需手动组织目录 |
| HOTA 等专业跟踪评估 | 尚未接入 trackeval 库 | 跟踪精度评估暂无 HOTA 指标 |
| 大规模批量评估（多序列汇总） | `validate_detector.py` 每次只处理一个序列 | 多序列需要循环调用 |
