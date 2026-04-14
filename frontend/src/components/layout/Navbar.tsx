/**
 * 顶部导航栏
 */
import { Link, NavLink, useLocation } from 'react-router-dom';
import { Activity, Video, Camera, LayoutDashboard, Wifi, WifiOff, Loader2 } from 'lucide-react';
import { useAppStore } from '../../store/appStore';

const NAV_LINKS = [
  { to: '/', label: '总控台', icon: LayoutDashboard },
  { to: '/upload', label: '视频预测', icon: Video },
  { to: '/camera', label: '摄像头预测', icon: Camera },
];

export default function Navbar() {
  const backendStatus = useAppStore((s) => s.backendStatus);
  const location = useLocation();

  const statusIndicator = () => {
    if (backendStatus === 'online') {
      return (
        <div className="flex items-center gap-1.5 text-emerald-400 text-sm">
          <Wifi className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">后端在线</span>
        </div>
      );
    }
    if (backendStatus === 'offline') {
      return (
        <div className="flex items-center gap-1.5 text-red-400 text-sm">
          <WifiOff className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">后端离线</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1.5 text-slate-500 text-sm">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        <span className="hidden sm:inline">连接中</span>
      </div>
    );
  };

  return (
    <nav className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="h-14 flex items-center justify-between gap-4">
          {/* 品牌 */}
          <Link to="/" className="flex items-center gap-2.5 shrink-0">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-white text-sm sm:text-base">
              EKF 跟踪系统
            </span>
          </Link>

          {/* 导航链接 */}
          <div className="flex items-center gap-0.5">
            {NAV_LINKS.map(({ to, label, icon: Icon }) => {
              const isActive =
                to === '/' ? location.pathname === '/' : location.pathname.startsWith(to);
              return (
                <NavLink
                  key={to}
                  to={to}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-600/20 text-blue-400'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{label}</span>
                </NavLink>
              );
            })}
          </div>

          {/* 后端状态 */}
          <div className="shrink-0">{statusIndicator()}</div>
        </div>
      </div>
    </nav>
  );
}
