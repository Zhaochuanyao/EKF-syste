/**
 * API 封装层
 * 所有与后端的通信均通过此模块进行
 */
import axios, { AxiosError } from 'axios';
import type {
  HealthResponse,
  FramePredictResponse,
  VideoTaskStartResponse,
  VideoTaskStatus,
  VideoUploadOptions,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 60_000,
});

/** 统一错误处理，提取后端 detail 消息 */
function extractError(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (err.response?.status === 0 || err.code === 'ERR_NETWORK') {
      return '无法连接后端服务，请确认已启动 uvicorn';
    }
    return err.message;
  }
  return String(err);
}

// ── 健康检查 ──────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health', { timeout: 5_000 });
  return data;
}

// ── 单帧预测（摄像头模式）────────────────────────────────────

export async function predictFrame(
  imageBase64: string,
  frameId: number = 0,
): Promise<FramePredictResponse> {
  try {
    const { data } = await client.post<FramePredictResponse>('/predict/frame', {
      image_base64: imageBase64,
      frame_id: frameId,
    });
    return data;
  } catch (err) {
    throw new Error(extractError(err));
  }
}

// ── 视频任务（上传 + 处理）───────────────────────────────────

export async function startVideoTask(
  file: File,
  options: VideoUploadOptions = {},
): Promise<VideoTaskStartResponse> {
  const {
    showTracks = true,
    showFuture = true,
    frameSkip = 1,
    configName = 'base',
    onUploadProgress,
  } = options;

  const params = new URLSearchParams({
    show_tracks: String(showTracks),
    show_future: String(showFuture),
    frame_skip: String(frameSkip),
    config_name: configName,
  });

  const formData = new FormData();
  formData.append('file', file);

  try {
    const { data } = await client.post<VideoTaskStartResponse>(
      `/predict/video/start?${params.toString()}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: onUploadProgress
          ? (evt) => {
              if (evt.total) {
                onUploadProgress(Math.round((evt.loaded / evt.total) * 100));
              }
            }
          : undefined,
      },
    );
    return data;
  } catch (err) {
    throw new Error(extractError(err));
  }
}

export async function getVideoStatus(taskId: string): Promise<VideoTaskStatus> {
  try {
    const { data } = await client.get<VideoTaskStatus>(
      `/predict/video/status/${taskId}`,
    );
    return data;
  } catch (err) {
    throw new Error(extractError(err));
  }
}

/** 构造输出文件的完整 URL */
export function getOutputUrl(filename: string): string {
  return `${API_BASE}/outputs/${filename}`;
}

export async function resetTracker(): Promise<void> {
  await client.post('/reset');
}
