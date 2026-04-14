/**
 * 状态标签徽章
 */

type StatusType = 'success' | 'error' | 'warning' | 'info' | 'processing' | 'idle';

interface StatusBadgeProps {
  status: StatusType;
  label: string;
  pulse?: boolean;
}

const CONFIG: Record<StatusType, { dot: string; text: string; bg: string }> = {
  success:    { dot: 'bg-emerald-400', text: 'text-emerald-400', bg: 'bg-emerald-400/10 border-emerald-400/20' },
  error:      { dot: 'bg-red-400',     text: 'text-red-400',     bg: 'bg-red-400/10 border-red-400/20' },
  warning:    { dot: 'bg-amber-400',   text: 'text-amber-400',   bg: 'bg-amber-400/10 border-amber-400/20' },
  info:       { dot: 'bg-blue-400',    text: 'text-blue-400',    bg: 'bg-blue-400/10 border-blue-400/20' },
  processing: { dot: 'bg-cyan-400',    text: 'text-cyan-400',    bg: 'bg-cyan-400/10 border-cyan-400/20' },
  idle:       { dot: 'bg-slate-500',   text: 'text-slate-400',   bg: 'bg-slate-800 border-slate-700' },
};

export default function StatusBadge({ status, label, pulse }: StatusBadgeProps) {
  const c = CONFIG[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${c.bg} ${c.text}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot} ${pulse ? 'animate-pulse' : ''}`} />
      {label}
    </span>
  );
}
