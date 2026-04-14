/**
 * 通用加载指示器
 */
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
}

const SIZES = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
};

export default function LoadingSpinner({ text, size = 'md' }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 text-slate-400">
      <Loader2 className={`${SIZES[size]} animate-spin text-blue-400`} />
      {text && <p className="text-sm">{text}</p>}
    </div>
  );
}
