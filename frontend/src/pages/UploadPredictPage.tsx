import { useState, useRef, useEffect } from 'react';
import {
  Upload, FileVideo, Settings2, Play, Download, RefreshCw,
  CheckCircle2, XCircle, AlertCircle, Clock, Film, Target,
  BarChart3, Loader2, X, Activity,
} from 'lucide-react';
import { startVideoTask, getVideoStatus, getOutputUrl } from '../services/api';
import NoiseModePicker from '../components/common/NoiseModePicker';
import type { VideoResult } from '../services/types';

type Phase = 'idle' | 'selected' | 'uploading' | 'processing' | 'success' | 'error';

interface PageState {
  phase: Phase;
  file: File | null;
  uploadPct: number;
  processPct: number;
  taskId: string | null;
  result: VideoResult | null;
  errorMsg: string | null;
  videoError: boolean;
}

interface Options {
  showTracks: boolean;
  showFuture: boolean;
  frameSkip: number;
  configName: string;
}

const ACCEPTED = ['.mp4', '.avi', '.mov', '.mkv'];

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDuration(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}分${s}秒` : `${s}秒`;
}

function DropZone({ onFile }: { onFile: (f: File) => void }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files?.length) return;
    const f = files[0];
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      alert(`不支持的文件格式，请上传 ${ACCEPTED.join(' / ')}`);
      return;
    }
    onFile(f);
  };

  return (
    <div
      onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { e.preventDefault(); setDragging(false); }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-xl border-2 border-dashed p-8 flex flex-col items-center gap-3 transition-colors ${
        dragging ? 'border-blue-400 bg-blue-500/10' : 'border-slate-700 hover:border-slate-500 bg-slate-900/40 hover:bg-slate-900/70'
      }`}
    >
      <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-colors ${dragging ? 'bg-blue-500/20' : 'bg-slate-800'}`}>
        <Upload className={`w-5 h-5 ${dragging ? 'text-blue-400' : 'text-slate-400'}`} />
      </div>
      <div className="text-center">
        <p className="text-slate-200 font-medium text-sm">拖拽视频文件到此处</p>
        <p className="text-slate-500 text-xs mt-1">或点击选择文件</p>
        <p className="text-slate-600 text-xs mt-2">支持 MP4 · AVI · MOV · MKV</p>
      </div>
      <input ref={inputRef} type="file" accept={ACCEPTED.join(',')} className="hidden" onChange={(e) => handleFiles(e.target.files)} />
    </div>
  );
}

function ProgressBar({ value, label, color = 'bg-blue-500' }: { value: number; label: string; color?: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{label}</span><span>{value}%</span>
      </div>
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-300 progress-stripe`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

const PHASE_LABEL: Record<Phase, string> = {
  idle: '未开始', selected: '已选择文件', uploading: '上传中',
  processing: '推理中', success: '已完成', error: '处理失败',
};
const PHASE_COLOR: Record<Phase, string> = {
  idle: 'text-slate-500', selected: 'text-blue-400', uploading: 'text-blue-400',
  processing: 'text-cyan-400', success: 'text-emerald-400', error: 'text-red-400',
};

export default function UploadPredictPage() {
  const [state, setState] = useState<PageState>({
    phase: 'idle', file: null, uploadPct: 0, processPct: 0,
    taskId: null, result: null, errorMsg: null, videoError: false,
  });
  const [options, setOptions] = useState<Options>({
    showTracks: true, showFuture: true, frameSkip: 1, configName: 'demo_vehicle_accuracy',
  });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  useEffect(() => () => stopPolling(), []);

  const handleFile = (f: File) => setState((s) => ({ ...s, phase: 'selected', file: f, result: null, errorMsg: null }));

  const handleClearFile = () => {
    stopPolling();
    setState({ phase: 'idle', file: null, uploadPct: 0, processPct: 0, taskId: null, result: null, errorMsg: null, videoError: false });
  };

  const handleSubmit = async () => {
    if (!state.file) return;
    setState((s) => ({ ...s, phase: 'uploading', uploadPct: 0, errorMsg: null }));
    try {
      const taskResp = await startVideoTask(state.file, {
        ...options,
        onUploadProgress: (pct) => setState((s) => ({ ...s, uploadPct: pct })),
      });
      setState((s) => ({ ...s, phase: 'processing', processPct: 0, taskId: taskResp.task_id }));
      pollRef.current = setInterval(async () => {
        try {
          const status = await getVideoStatus(taskResp.task_id);
          setState((s) => ({ ...s, processPct: status.progress }));
          if (status.status === 'done' && status.result) {
            stopPolling();
            setState((s) => ({ ...s, phase: 'success', result: status.result }));
          } else if (status.status === 'error') {
            stopPolling();
            setState((s) => ({ ...s, phase: 'error', errorMsg: status.error || '处理失败' }));
          }
        } catch (err) {
          stopPolling();
          setState((s) => ({ ...s, phase: 'error', errorMsg: String(err) }));
        }
      }, 2000);
    } catch (err) {
      setState((s) => ({ ...s, phase: 'error', errorMsg: String(err) }));
    }
  };

  const isProcessing = state.phase === 'uploading' || state.phase === 'processing';

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <div className="mb-7">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2.5">
          <FileVideo className="w-6 h-6 text-blue-400" />
          视频上传预测
        </h1>
        <p className="text-slate-500 mt-1 text-sm">上传交通监控视频，批量检测并跟踪车辆，生成带轨迹标注的结果视频</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* 左侧面板 */}
        <div className="lg:col-span-2 space-y-5">

          {/* 上传卡片 */}
          <div className="card p-5 space-y-4">
            <h2 className="card-header"><Upload className="w-4 h-4 text-blue-400" />上传文件</h2>
            {state.file ? (
              <div className="flex items-start gap-3 p-3 bg-slate-800/60 rounded-xl border border-slate-700">
                <FileVideo className="w-7 h-7 text-blue-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-slate-200 text-sm font-medium truncate">{state.file.name}</p>
                  <p className="text-slate-500 text-xs mt-0.5">{formatBytes(state.file.size)}</p>
                </div>
                {!isProcessing && (
                  <button onClick={handleClearFile} className="shrink-0 p-1 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-700 transition-colors">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            ) : (
              <DropZone onFile={handleFile} />
            )}
          </div>

          {/* 处理参数 */}
          <div className="card p-5 space-y-4">
            <h2 className="card-header"><Settings2 className="w-4 h-4 text-slate-400" />处理参数</h2>
            <div>
              <label className="label">处理模式</label>
              <select value={options.configName} onChange={(e) => setOptions((o) => ({ ...o, configName: e.target.value }))} className="input w-full text-sm" disabled={isProcessing}>
                <option value="demo_vehicle_accuracy">车辆稳定模式（默认）</option>
                <option value="base">自动（通用）</option>
                <option value="demo_person_accuracy">行人精度模式</option>
              </select>
            </div>
            <NoiseModePicker isRunning={isProcessing} />
            <div>
              <label className="label">跳帧间隔</label>
              <select value={options.frameSkip} onChange={(e) => setOptions((o) => ({ ...o, frameSkip: Number(e.target.value) }))} className="input w-full text-sm" disabled={isProcessing}>
                {[1, 2, 3, 5].map((v) => <option key={v} value={v}>每 {v} 帧处理一次</option>)}
              </select>
            </div>
            <div className="space-y-3 pt-1">
              {([['showTracks', '显示轨迹框 & ID'], ['showFuture', '显示预测轨迹点']] as [keyof Options, string][]).map(([key, label]) => (
                <label key={key} className="flex items-center justify-between cursor-pointer group">
                  <span className="text-sm text-slate-400 group-hover:text-slate-300 transition-colors">{label}</span>
                  <button type="button" role="switch" aria-checked={options[key] as boolean} disabled={isProcessing}
                    onClick={() => setOptions((o) => ({ ...o, [key]: !o[key] }))}
                    className={`relative w-9 h-5 rounded-full transition-colors ${options[key] ? 'bg-blue-600' : 'bg-slate-700'} disabled:opacity-50`}>
                    <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${options[key] ? 'translate-x-4' : 'translate-x-0'}`} />
                  </button>
                </label>
              ))}
            </div>
          </div>

          {/* 运行状态卡片 */}
          <div className="card p-5 space-y-3">
            <h2 className="card-header"><Activity className="w-4 h-4 text-slate-400" />运行状态</h2>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-500">当前状态</span>
              <span className={`font-medium ${PHASE_COLOR[state.phase]}`}>{PHASE_LABEL[state.phase]}</span>
            </div>
            {(state.phase === 'uploading' || state.phase === 'processing') && (
              <ProgressBar
                value={state.phase === 'uploading' ? state.uploadPct : state.processPct}
                label={state.phase === 'uploading' ? '文件上传' : '视频处理'}
                color={state.phase === 'uploading' ? 'bg-blue-500' : 'bg-cyan-500'}
              />
            )}
            {state.phase === 'error' && state.errorMsg && (
              <p className="text-red-400 text-xs flex items-start gap-1.5">
                <XCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />{state.errorMsg}
              </p>
            )}
          </div>

          {/* 提交 / 重置 */}
          {state.phase !== 'success' ? (
            <button onClick={handleSubmit} disabled={!state.file || isProcessing} className="btn-primary w-full py-3 text-base">
              {isProcessing ? <><Loader2 className="w-4 h-4 animate-spin" />处理中...</> : <><Play className="w-4 h-4" />开始处理</>}
            </button>
          ) : (
            <button onClick={handleClearFile} className="btn-secondary w-full py-3 text-base">
              <RefreshCw className="w-4 h-4" />重新上传
            </button>
          )}
        </div>

        {/* 右侧结果区 */}
        <div className="lg:col-span-3 space-y-5">

          {/* 成功：视频播放器 */}
          {state.phase === 'success' && state.result && (
            <div className="space-y-4 animate-slide-up">
              <div className="flex items-center gap-2.5 px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400 text-sm">
                <CheckCircle2 className="w-4 h-4 shrink-0" />视频处理完成！
              </div>
              <div className="card overflow-hidden">
                <div className="bg-black aspect-video flex items-center justify-center relative">
                  {state.videoError ? (
                    <div className="flex flex-col items-center gap-3 text-center px-6">
                      <AlertCircle className="w-10 h-10 text-amber-400" />
                      <div>
                        <p className="text-amber-400 font-medium">视频无法在浏览器中播放</p>
                        <p className="text-slate-500 text-sm mt-1">请点击下方「下载视频」保存后用本地播放器打开</p>
                      </div>
                    </div>
                  ) : (
                    <video key={state.result.output_file} controls className="w-full h-full object-contain" preload="auto"
                      onError={() => setState((s) => ({ ...s, videoError: true }))}>
                      <source src={getOutputUrl(state.result.output_file)} type="video/mp4" />
                      您的浏览器不支持视频播放
                    </video>
                  )}
                </div>
                <div className="p-4 flex items-center justify-between border-t border-slate-800">
                  <span className="text-slate-400 text-sm font-mono truncate max-w-xs">{state.result.output_file}</span>
                  <a href={getOutputUrl(state.result.output_file)} download={state.result.output_file} className="btn-primary text-sm py-2">
                    <Download className="w-3.5 h-3.5" />下载视频
                  </a>
                </div>
              </div>

              {/* 统计摘要 */}
              <div className="card p-5">
                <h3 className="card-header mb-4"><BarChart3 className="w-4 h-4 text-blue-400" />处理摘要</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {[
                    { icon: Film,     label: '处理帧数',   value: state.result.frames_processed.toLocaleString() },
                    { icon: Target,   label: '检测次数',   value: state.result.total_detections.toLocaleString() },
                    { icon: Activity, label: '轨迹数',     value: state.result.unique_tracks.toString() },
                    { icon: Clock,    label: '视频时长',   value: formatDuration(state.result.duration_seconds) },
                  ].map(({ icon: Icon, label, value }) => (
                    <div key={label} className="bg-slate-800/60 rounded-lg p-3 text-center">
                      <Icon className="w-4 h-4 text-slate-500 mx-auto mb-1.5" />
                      <p className="text-lg font-bold text-slate-100">{value}</p>
                      <p className="text-slate-500 text-xs mt-0.5">{label}</p>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 gap-3 text-sm">
                  <div className="flex justify-between text-slate-400">
                    <span>视频帧率</span><span className="text-slate-300">{state.result.fps} fps</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>处理模式</span><span className="text-slate-300">{options.configName}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 空状态 / 处理中 */}
          {state.phase !== 'success' && (
            <div className="card p-12 flex flex-col items-center justify-center text-center gap-4 min-h-[320px]">
              {isProcessing ? (
                <>
                  <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
                  <div>
                    <p className="text-slate-300 font-medium">
                      {state.phase === 'uploading' ? '正在上传文件...' : '正在逐帧推理，请稍候...'}
                    </p>
                    <p className="text-slate-600 text-sm mt-1">
                      {state.phase === 'processing' ? '检测 → 跟踪 → EKF 预测 → 可视化' : '上传完成后将自动开始处理'}
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center">
                    <Film className="w-7 h-7 text-slate-600" />
                  </div>
                  <div>
                    <p className="text-slate-400 font-medium">结果将在此处显示</p>
                    <p className="text-slate-600 text-sm mt-1">
                      {state.phase === 'idle' ? '请先在左侧上传视频文件' : '已选择文件，配置参数后点击「开始处理」'}
                    </p>
                  </div>
                  {state.phase === 'error' && state.errorMsg && (
                    <div className="flex items-start gap-2 text-red-400 text-sm max-w-sm">
                      <XCircle className="w-4 h-4 shrink-0 mt-0.5" />{state.errorMsg}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
