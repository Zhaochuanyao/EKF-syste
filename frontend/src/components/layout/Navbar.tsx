import { Link, NavLink, useLocation } from 'react-router-dom';
import { Activity, Video, Camera, LayoutDashboard } from 'lucide-react';
import { useAppStore } from '../../store/appStore';

const NAV_LINKS = [
  { to: '/', label: '总控台', icon: LayoutDashboard },
  { to: '/upload', label: '视频预测', icon: Video },
  { to: '/camera', label: '摄像头预测', icon: Camera },
];

function BackendDot() {
  const status = useAppStore((s) => s.backendStatus);
  if (status === 'online') {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
        </span>
        <span className="text-emerald-400 hidden sm:inline">后端在线</span>
      </div>
    );
  }
  if (status === 'offline') {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="inline-flex rounded-full h-2 w-2 bg-red-500" />
        <span className="text-red-400 hidden sm:inline">后端离线</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="inline-flex rounded-full h-2 w-2 bg-slate-600 animate-pulse" />
      <span className="text-slate-500 hidden sm:inline">连接中</span>
    </div>
  );
}

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="h-14 flex items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-2.5 shrink-0">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-white text-sm sm:text-base">EKF 多目标跟踪系统</span>
          </Link>

          <div className="flex items-center gap-0.5">
            {NAV_LINKS.map(({ to, label, icon: Icon }) => {
              const isActive = to === '/' ? location.pathname === '/' : location.pathname.startsWith(to);
              return (
                <NavLink
                  key={to}
                  to={to}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive ? 'bg-blue-600/20 text-blue-400' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{label}</span>
                </NavLink>
              );
            })}
          </div>

          <div className="shrink-0"><BackendDot /></div>
        </div>
      </div>
    </nav>
  );
}
