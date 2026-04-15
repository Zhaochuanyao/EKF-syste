/** API 数据类型定义 */

export interface HealthResponse {
  status: string;
  version?: string;
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrackInfo {
  track_id: number;
  bbox: BBox;
  score: number;
  class_id: number;
  class_name: string;
  /** Tentative | Confirmed | Lost */
  state: string;
  center: [number, number];
  future_points: Record<number, [number, number]>;
  /** EKF 滤波后原始历史轨迹 */
  raw_history?: [number, number][];
  /** EMA 平滑后历史轨迹 */
  smoothed_history?: [number, number][];
  /** EKF 估计速度（像素/秒）*/
  velocity?: number;
  /** EKF 估计航向（弧度）*/
  heading?: number;
  /** EKF 估计角速度（弧度/秒）*/
  omega?: number;
  /** 是否处于 Lost 恢复保护期 */
  recovered_recently?: boolean;
}

export interface FramePredictResponse {
  frame_id: number;
  tracks: TrackInfo[];
  num_detections: number;
  process_time_ms: number;
}

export interface VideoTaskStartResponse {
  task_id: string;
  status: string;
}

export interface VideoResult {
  output_file: string;
  frames_processed: number;
  total_detections: number;
  unique_tracks: number;
  fps: number;
  duration_seconds: number;
}

export interface VideoTaskStatus {
  status: 'processing' | 'done' | 'error';
  progress: number;
  result: VideoResult | null;
  error: string | null;
  filename: string;
  config_name?: string;
}

export interface VideoUploadOptions {
  showTracks?: boolean;
  showFuture?: boolean;
  frameSkip?: number;
  configName?: string;
  onUploadProgress?: (percent: number) => void;
}

export interface RealtimeStats {
  trackCount: number;
  confirmedCount: number;
  tentativeCount: number;
  detectionCount: number;
  inferenceMs: number;
  frameId: number;
}
