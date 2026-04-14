"""
FastAPI 服务接口
"""

import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2

from .schemas import (
    FramePredictRequest,
    FramePredictResponse,
    HealthResponse,
    TrackInfo,
    BBox,
)
from .service import TrackingService
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.serving.api")

app = FastAPI(
    title="EKF 多目标跟踪 API",
    description="基于扩展卡尔曼滤波的目标检测与运动轨迹预测系统",
    version="1.0.0",
)

# ── CORS（允许前端开发服务器跨域访问）──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 输出目录 ──────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── 异步任务注册表 ─────────────────────────────────────────────
_tasks: Dict[str, Dict] = {}

# ── 全局服务实例（懒加载）─────────────────────────────────────
_service: Optional[TrackingService] = None


def get_service() -> TrackingService:
    global _service
    if _service is None:
        logger.info("初始化跟踪服务...")
        _service = TrackingService()
    return _service


# ══════════════════════════════════════════════════════════════
# 健康检查
# ══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(status="ok")


# ══════════════════════════════════════════════════════════════
# 单帧预测（摄像头实时模式）
# ══════════════════════════════════════════════════════════════

@app.post("/predict/frame", response_model=FramePredictResponse)
async def predict_frame(request: FramePredictRequest):
    """
    单帧预测接口。
    请求体: { "image_base64": "<base64图像>", "frame_id": 0 }
    """
    service = get_service()
    try:
        frame = service.decode_base64_image(request.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="图像解码失败")

        results, num_dets, elapsed_ms = service.process_frame(frame)

        tracks = [
            TrackInfo(
                track_id=r["track_id"],
                bbox=BBox(**r["bbox"]),
                score=r["score"],
                class_id=r["class_id"],
                class_name=r["class_name"],
                state=r["state"],
                center=r["center"],
                future_points={int(k): v for k, v in r["future_points"].items()},
            )
            for r in results
        ]

        return FramePredictResponse(
            frame_id=request.frame_id,
            tracks=tracks,
            num_detections=num_dets,
            process_time_ms=round(elapsed_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"单帧预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# 视频预测（异步后台任务）
# ══════════════════════════════════════════════════════════════

@app.post("/predict/video/start")
async def predict_video_start(
    file: UploadFile = File(...),
    show_tracks: bool = Query(default=True, description="是否绘制轨迹框"),
    show_future: bool = Query(default=True, description="是否绘制预测点"),
    frame_skip: int = Query(default=1, ge=1, le=10, description="跳帧间隔"),
    config_name: str = Query(default="base", description="配置名称"),
):
    """
    启动异步视频处理任务。
    返回 task_id，通过 GET /predict/video/status/{task_id} 轮询进度。
    """
    task_id = uuid.uuid4().hex[:8]
    original_name = file.filename or "video.mp4"
    suffix = Path(original_name).suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".mkv"):
        suffix = ".mp4"

    tmp_path = os.path.join(tempfile.gettempdir(), f"ekf_upload_{task_id}{suffix}")
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    logger.info(f"任务 {task_id}: 文件={original_name}, 配置={config_name}, 跳帧={frame_skip}")
    _tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "result": None,
        "error": None,
        "filename": original_name,
        "config_name": config_name,
    }

    def _run_task():
        try:
            svc = get_service()
            result = svc.process_video_to_file(
                video_path=tmp_path,
                task_id=task_id,
                show_tracks=show_tracks,
                show_future=show_future,
                frame_skip=frame_skip,
                tasks=_tasks,
            )
            _tasks[task_id]["status"] = "done"
            _tasks[task_id]["progress"] = 100
            _tasks[task_id]["result"] = result
            logger.info(f"视频任务 {task_id} 完成: {result}")
        except Exception as exc:
            logger.error(f"视频任务 {task_id} 失败: {exc}", exc_info=True)
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    threading.Thread(target=_run_task, daemon=True).start()
    return {"task_id": task_id, "status": "processing"}


@app.get("/predict/video/status/{task_id}")
async def predict_video_status(task_id: str):
    """查询视频处理任务进度"""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _tasks[task_id]


# ══════════════════════════════════════════════════════════════
# 输出文件服务
# ══════════════════════════════════════════════════════════════

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """获取输出文件（视频等）"""
    safe_name = Path(filename).name  # 防止路径穿越
    path = OUTPUTS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    media_type = "video/mp4" if safe_name.endswith(".mp4") else "application/octet-stream"
    return FileResponse(
        str(path),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{safe_name}"',
        },
    )


# ══════════════════════════════════════════════════════════════
# 重置跟踪器
# ══════════════════════════════════════════════════════════════

@app.post("/reset")
async def reset_tracker():
    """重置跟踪器状态"""
    service = get_service()
    service.tracker.reset()
    service._frame_id = 0
    return {"status": "reset ok"}


# ══════════════════════════════════════════════════════════════
# 旧版同步接口（兼容保留）
# ══════════════════════════════════════════════════════════════

@app.post("/predict/video")
async def predict_video_legacy(file: UploadFile = File(...)):
    """旧版同步视频接口（不生成输出视频，仅返回统计）"""
    service = get_service()
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"ekf_legacy_{uuid.uuid4().hex[:6]}{suffix}"
    )
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    try:
        cap = cv2.VideoCapture(tmp_path)
        frame_count, total_tracks = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results, _, _ = service.process_frame(frame)
            total_tracks += len(results)
            frame_count += 1
        cap.release()
        return JSONResponse({
            "status": "ok",
            "frames_processed": frame_count,
            "total_track_instances": total_tracks,
        })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
