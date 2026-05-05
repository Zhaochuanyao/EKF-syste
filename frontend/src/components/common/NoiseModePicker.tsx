import { useState, useEffect } from 'react';
import { Sliders, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { getNoiseMode, updateNoiseMode, NOISE_MODE_OPTIONS } from '../../services/api';

const STRATEGY_DESC: Record<string, string> = {
  current_ekf: '固定基础 R/Q，不启用主动调节，标准 EKF 行为',
  r_adapt:     '仅启用测量噪声 R 自适应，抑制异常检测框影响',
  q_sched:     '仅启用过程噪声 Q 机动调度，适应转弯与变速场景',
  rq_adapt:    '同时启用 R 自适应与 Q 调度，不启用鲁棒兜底',
  full_adpt:   '启用 R/Q 自适应与异常观测鲁棒处理，完整自适应模式',
};

interface Props {
  isRunning?: boolean;
}

export default function NoiseModePicker({ isRunning = false }: Props) {
  const [currentMode, setCurrentMode] = useState<string>('current_ekf');
  const [switching, setSwitching] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getNoiseMode()
      .then((info) => setCurrentMode(info.mode))
      .catch(() => {});
  }, []);

  const handleChange = async (mode: string) => {
    if (mode === currentMode || switching) return;
    setSwitching(true);
    setError(null);
    setHint(null);
    try {
      await updateNoiseMode(mode);
      setCurrentMode(mode);
      setHint(isRunning ? '新策略将从下一帧开始生效' : '策略已切换');
      setTimeout(() => setHint(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : '切换失败，请检查后端连接');
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-blue-400" />
          <span className="label mb-0">EKF 噪声策略</span>
        </div>
        {switching && <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />}
      </div>

      <p className="text-slate-600 text-xs">用于切换轨迹预测中的 R/Q 自适应调节方式</p>

      {/* 策略选项 */}
      <div className="space-y-1.5">
        {NOISE_MODE_OPTIONS.map(({ value, label }) => {
          const isActive = currentMode === value;
          return (
            <button
              key={value}
              disabled={switching}
              onClick={() => handleChange(value)}
              className={[
                'w-full text-left px-3 py-2.5 rounded-lg border transition-all duration-150',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                isActive
                  ? 'bg-blue-600/15 border-blue-500/40 text-blue-300'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300',
              ].join(' ')}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold font-mono">{label}</span>
                {isActive && <span className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />}
              </div>
              <p className="text-xs mt-0.5 opacity-70 leading-snug">{STRATEGY_DESC[value]}</p>
            </button>
          );
        })}
      </div>

      {hint && (
        <p className="text-xs text-emerald-400 flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />{hint}
        </p>
      )}
      {error && (
        <p className="text-xs text-red-400 flex items-center gap-1.5">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />{error}
        </p>
      )}
    </div>
  );
}
