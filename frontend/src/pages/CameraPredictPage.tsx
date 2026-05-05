import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Camera, CameraOff, Play, Pause, Square, Settings2,
  AlertTriangle, Loader2, Eye, EyeOff,
  Activity, Target, Zap, Monitor, Wifi,
} from 'lucide-react';
import { predictFrame, resetTracker } from '../services/api';
import NoiseModePicker from '../components/common/NoiseModePicker';
import type { TrackInfo, RealtimeStats } from '../services/types';
import { useCamera } from '../hooks/useCamera';
import StatusBadge from '../components/common/StatusBadge';
import { useAppStore } from '../store/appStore';

const TRACK_COLORS = ['#3b82f6','#ef4444','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#f97316','#ec4899','#14b8a6','#84cc16'];
const getTrackColor = (id: number) => TRACK_COLORS[id % TRACK_COLORS.length];

function drawOverlay(
  canvas: HTMLCanvasElement, tracks: TrackInfo[],
  videoW: number, videoH: number,
  showTracks: boolean, showFuture: boolean, showSmoothed: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dw = canvas.width, dh = canvas.height;
  ctx.clearRect(0, 0, dw, dh);
  if (!videoW || !videoH) return;
  const sx = dw / videoW, sy = dh / videoH;

  tracks.forEach((track) => {
    const color = getTrackColor(track.track_id);
    const { x1, y1, x2, y2 } = track.bbox;
    const bx = x1 * sx, by = y1 * sy, bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;

    if (showTracks) {
      const history = showSmoothed
        ? (track.smoothed_history ?? track.raw_history ?? [])
        : (track.raw_history ?? []);
      if (history.length >= 2) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(history[0][0] * sx, history[0][1] * sy);
        for (let i = 1; i < history.length; i++) ctx.lineTo(history[i][0] * sx, history[i][1] * sy);
        ctx.stroke(); ctx.globalAlpha = 1.0;
      }
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);
      const label = `#${track.track_id} ${track.class_name}`;
      ctx.font = 'bold 12px "JetBrains Mono", monospace';
      const tw = ctx.measureText(label).width + 10, th = 18;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.roundRect(bx - 1, by - th - 2, tw, th, 4); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.fillText(label, bx + 4, by - 6);
      ctx.font = '10px sans-serif'; ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillText(`${Math.round(track.score * 100)}%`, bx + bw - 28, by + 14);
    }

    if (showFuture) {
      const fps = Object.entries(track.future_points).sort(([a], [b]) => Number(a) - Number(b));
      if (fps.length > 0) {
        const pts = fps.map(([, p]) => [p[0] * sx, p[1] * sy]);
        const cx = (x1 + x2) / 2 * sx, cy = (y1 + y2) / 2 * sy;
        ctx.strokeStyle = 'rgba(6,182,212,0.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(cx, cy);
        pts.forEach(([px, py]) => ctx.lineTo(px, py));
        ctx.stroke(); ctx.setLineDash([]);
        pts.forEach(([px, py], i) => {
          ctx.fillStyle = `rgba(6,182,212,${1 - i * 0.2})`;
          ctx.beginPath(); ctx.arc(px, py, Math.max(2, 5 - i), 0, Math.PI * 2); ctx.fill();
        });
      }
    }
  });
}

type PredictPhase = 'idle' | 'predicting' | 'paused';

export default function CameraPredictPage() {
  const { state: camState, videoRef, startCamera, stopCamera, selectDevice } = useCamera();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const backendStatus = useAppStore((s) => s.backendStatus);

  const [predictPhase, setPredictPhase] = useState<PredictPhase>('idle');
  const [stats, setStats] = useState<RealtimeStats>({
    trackCount: 0, confirmedCount: 0, tentativeCount: 0,
    detectionCount: 0, inferenceMs: 0, frameId: 0,
  });
  const [showTracks, setShowTracks] = useState(true);
  const [showFuture, setShowFuture] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [fps, setFps] = useState(5);
  const [configName, setConfigName] = useState('demo_vehicle_accuracy');

  const frameIdRef = useRef(0);
  const isPredictingRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    if (video.videoWidth && video.videoHeight) {
      canvas.width = video.offsetWidth || video.videoWidth;
      canvas.height = video.offsetHeight || video.videoHeight;
    }
  }, [videoRef]);

  useEffect(() => {
    const obs = new ResizeObserver(syncCanvasSize);
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [syncCanvasSize]);

  const predictOneFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || isPredictingRef.current || video.readyState < 2 || !video.videoWidth) return;
    isPredictingRef.current = true;
    try {
      const cap = document.createElement('canvas');
      cap.width = video.videoWidth; cap.height = video.videoHeight;
      cap.getContext('2d')!.drawImage(video, 0, 0);
      const b64 = cap.toDataURL('image/jpeg', 0.7).split(',')[1];
      const resp = await predictFrame(b64, frameIdRef.current++, configName);
      if (canvasRef.current) {
        syncCanvasSize();
        drawOverlay(canvasRef.current, resp.tracks, video.videoWidth, video.videoHeight, showTracks, showFuture, showSmoothed);
      }
      setStats({
        trackCount: resp.tracks.length,
        confirmedCount: resp.tracks.filter((t) => t.state === 'Confirmed').length,
        tentativeCount: resp.tracks.filter((t) => t.state === 'Tentative').length,
        detectionCount: resp.num_detections,
        inferenceMs: Math.round(resp.process_time_ms),
        frameId: frameIdRef.current,
      });
    } catch { /* 静默忽略单帧失败 */ } finally {
      isPredictingRef.current = false;
    }
  }, [videoRef, showTracks, showFuture, showSmoothed, syncCanvasSize, configName]);

  useEffect(() => {
    if (predictPhase === 'predicting') {
      intervalRef.current = setInterval(predictOneFrame, Math.round(1000 / fps));
    } else {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [predictPhase, fps, predictOneFrame]);

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };
  const resetStats = () => setStats({ trackCount: 0, confirmedCount: 0, tentativeCount: 0, detectionCount: 0, inferenceMs: 0, frameId: 0 });

  const handleStartCamera = () => startCamera();
  const handleStopCamera = () => { setPredictPhase('idle'); stopCamera(); clearCanvas(); resetStats(); };
  const handleStartPredict = async () => {
    try { await resetTracker(); } catch { /* 非致命 */ }
    frameIdRef.current = 0;
    setPredictPhase('predicting');
  };
  const handleStop = () => { setPredictPhase('idle'); clearCanvas(); resetStats(); };

  const cameraActive = camState.status === 'active';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2.5">
          <Camera className="w-6 h-6 text-cyan-400" />摄像头实时预测
        </h1>
        <p className="text-slate-500 mt-1 text-sm">实时调用摄像头，对画面中车辆进行跟踪与短时轨迹预测</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* 左侧：摄像头画面 */}
        <div className="xl:col-span-3 space-y-4">
          <div className="card overflow-hidden">
            <div ref={containerRef} className="relative bg-black aspect-video flex items-center justify-center overflow-hidden">
              <video ref={videoRef} autoPlay muted playsInline onLoadedMetadata={syncCanvasSize}
                className={`w-full h-full object-contain ${cameraActive ? 'block' : 'hidden'}`} />
              <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" style={{ width: '100%', height: '100%' }} />

              {!cameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center">
                  {camState.status === 'requesting' ? (
                    <><Loader2 className="w-10 h-10 text-cyan-400 animate-spin" /><p className="text-slate-400">正在请求摄像头权限...</p></>
                  ) : camState.status === 'error' ? (
                    <><AlertTriangle className="w-10 h-10 text-amber-400" />
                      <div><p className="text-amber-400 font-medium">摄像头错误</p><p className="text-slate-500 text-sm mt-1 max-w-xs">{camState.error}</p></div></>
                  ) : (
                    <><div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center"><CameraOff className="w-7 h-7 text-slate-600" /></div>
                      <div><p className="text-slate-400 font-medium">摄像头未开启</p><p className="text-slate-600 text-sm mt-1">点击右侧「打开摄像头」开始</p></div></>
                  )}
                </div>
              )}

              {cameraActive && (
                <div className="absolute top-3 left-3 flex gap-2">
                  {predictPhase === 'predicting' && <StatusBadge status="success" label="推理中" pulse />}
                  {predictPhase === 'paused'     && <StatusBadge status="warning" label="已暂停" />}
                  {predictPhase === 'idle'       && <StatusBadge status="info"    label="摄像头已开启" />}
                </div>
              )}
              {predictPhase === 'predicting' && (
                <div className="absolute bottom-3 right-3 bg-black/60 px-2 py-1 rounded text-xs font-mono text-cyan-400">
                  {fps} fps · #{stats.frameId}
                </div>
              )}
            </div>
          </div>

          {/* 实时统计（摄像头开启后显示） */}
          {cameraActive && (
            <div className="grid grid-cols-4 gap-3">
              {[
                { icon: Target,   label: '车辆轨迹', value: stats.trackCount,     color: 'bg-blue-500/15 text-blue-400' },
                { icon: Activity, label: '已确认',   value: stats.confirmedCount, color: 'bg-emerald-500/15 text-emerald-400' },
                { icon: Eye,      label: '检测数',   value: stats.detectionCount, color: 'bg-cyan-500/15 text-cyan-400' },
                { icon: Zap,      label: '推理延迟', value: `${stats.inferenceMs}ms`, color: 'bg-amber-500/15 text-amber-400' },
              ].map(({ icon: Icon, label, value, color }) => (
                <div key={label} className="bg-slate-800/60 rounded-xl p-3 flex items-center gap-2.5">
                  <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center shrink-0`}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <p className="text-lg font-bold text-slate-100 leading-tight">{value}</p>
                    <p className="text-xs text-slate-500">{label}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 右侧：控制面板 */}
        <div className="xl:col-span-2 space-y-5">

          {/* 摄像头控制 */}
          <div className="card p-5 space-y-4">
            <h2 className="card-header"><Monitor className="w-4 h-4 text-slate-400" />摄像头控制</h2>
            {!cameraActive ? (
              <button onClick={handleStartCamera} disabled={camState.status === 'requesting'} className="btn-primary w-full py-3">
                {camState.status === 'requesting'
                  ? <><Loader2 className="w-4 h-4 animate-spin" />请求权限中...</>
                  : <><Camera className="w-4 h-4" />打开摄像头</>}
              </button>
            ) : (
              <div className="space-y-3">
                <div className="flex gap-2">
                  {predictPhase === 'idle'       && <button onClick={handleStartPredict} className="btn-primary flex-1 py-2.5"><Play className="w-4 h-4" />开始预测</button>}
                  {predictPhase === 'predicting' && <button onClick={() => setPredictPhase('paused')} className="btn-secondary flex-1 py-2.5"><Pause className="w-4 h-4" />暂停</button>}
                  {predictPhase === 'paused'     && <button onClick={() => setPredictPhase('predicting')} className="btn-primary flex-1 py-2.5"><Play className="w-4 h-4" />恢复</button>}
                  {predictPhase !== 'idle'       && <button onClick={handleStop} className="btn-secondary px-3 py-2.5"><Square className="w-4 h-4" /></button>}
                </div>
                <button onClick={handleStopCamera} className="btn-danger w-full py-2">
                  <CameraOff className="w-4 h-4" />关闭摄像头
                </button>
              </div>
            )}
            {camState.status === 'error' && (
              <p className="text-amber-400 text-xs flex items-start gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />{camState.error}
              </p>
            )}
          </div>

          {/* 参数配置 */}
          <div className="card p-5 space-y-4">
            <h2 className="card-header"><Settings2 className="w-4 h-4 text-slate-400" />参数配置</h2>
            {camState.devices.length > 0 && (
              <div>
                <label className="label">摄像头设备</label>
                <select value={camState.selectedDeviceId} onChange={(e) => selectDevice(e.target.value)} className="input w-full text-sm" disabled={predictPhase === 'predicting'}>
                  {camState.devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `摄像头 ${i + 1}`}</option>)}
                </select>
              </div>
            )}
            <div>
              <label className="label">处理模式</label>
              <select value={configName} onChange={(e) => setConfigName(e.target.value)} className="input w-full text-sm" disabled={predictPhase === 'predicting'}>
                <option value="demo_vehicle_accuracy">车辆稳定模式（默认）</option>
                <option value="base">自动（通用）</option>
                <option value="demo_person_accuracy">行人精度模式</option>
              </select>
            </div>
            <NoiseModePicker isRunning={predictPhase === 'predicting'} />
            <div>
              <label className="label">推理帧率（fps）</label>
              <select value={fps} onChange={(e) => setFps(Number(e.target.value))} className="input w-full text-sm" disabled={predictPhase === 'predicting'}>
                {[1, 2, 3, 5, 10].map((v) => <option key={v} value={v}>{v} fps（{Math.round(1000/v)}ms 间隔）</option>)}
              </select>
            </div>
            <div className="space-y-3 pt-1">
              {([
                ['showTracks',  '显示检测框 & ID',       showTracks,  setShowTracks],
                ['showFuture',  '显示预测轨迹点',         showFuture,  setShowFuture],
                ['showSmoothed','历史轨迹 EMA 平滑',      showSmoothed,setShowSmoothed],
              ] as [string, string, boolean, (v: boolean) => void][]).map(([key, label, value, setter]) => (
                <label key={key} className="flex items-center justify-between cursor-pointer group">
                  <span className="flex items-center gap-1.5 text-sm text-slate-400 group-hover:text-slate-300 transition-colors">
                    {value ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}{label}
                  </span>
                  <button type="button" role="switch" aria-checked={value} onClick={() => setter(!value)}
                    className={`relative w-9 h-5 rounded-full transition-colors ${value ? 'bg-cyan-600' : 'bg-slate-700'}`}>
                    <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${value ? 'translate-x-4' : 'translate-x-0'}`} />
                  </button>
                </label>
              ))}
            </div>
          </div>

          {/* 实时状态卡片 */}
          <div className="card p-5 space-y-3">
            <h2 className="card-header"><Activity className="w-4 h-4 text-slate-400" />实时状态</h2>
            {[
              { label: '后端连接', value: backendStatus === 'online' ? '已连接' : backendStatus === 'offline' ? '未连接' : '检测中',
                color: backendStatus === 'online' ? 'text-emerald-400' : backendStatus === 'offline' ? 'text-red-400' : 'text-slate-500' },
              { label: '摄像头',   value: camState.status === 'active' ? '已开启' : camState.status === 'requesting' ? '请求中' : camState.status === 'error' ? '出错' : '未开启',
                color: camState.status === 'active' ? 'text-emerald-400' : camState.status === 'error' ? 'text-red-400' : 'text-slate-500' },
              { label: '推理状态', value: predictPhase === 'predicting' ? '推理中' : predictPhase === 'paused' ? '已暂停' : '未开始',
                color: predictPhase === 'predicting' ? 'text-cyan-400' : predictPhase === 'paused' ? 'text-amber-400' : 'text-slate-500' },
              { label: '当前帧率', value: predictPhase === 'predicting' ? `${fps} fps` : '—', color: 'text-slate-300' },
              { label: '跟踪目标', value: cameraActive ? `${stats.trackCount} 个` : '—', color: 'text-slate-300' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex items-center justify-between text-sm">
                <span className="text-slate-500">{label}</span>
                <span className={`font-medium ${color}`}>{value}</span>
              </div>
            ))}
          </div>

          {/* 提示 */}
          <div className="card p-4 border-slate-800/80">
            <div className="flex items-start gap-2 text-slate-500 text-xs leading-relaxed">
              <Wifi className="w-3.5 h-3.5 shrink-0 mt-0.5 text-slate-600" />
              <div className="space-y-1">
                <p>每帧以 JPEG 格式发送后端推理，车辆轨迹与预测路径叠加绘制在画面上。</p>
                <p>后端负载较高时，建议降低推理帧率至 2~3 fps。</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
