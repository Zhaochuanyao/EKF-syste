/**
 * 页面布局容器
 */
import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import { useAppStore } from '../../store/appStore';
import { checkHealth } from '../../services/api';

export default function Layout() {
  const setBackendStatus = useAppStore((s) => s.setBackendStatus);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;

    const probe = async () => {
      try {
        await checkHealth();
        setBackendStatus('online');
      } catch {
        setBackendStatus('offline');
      }
    };

    probe();
    timer = setInterval(probe, 10_000); // 每 10 秒探测一次

    return () => clearInterval(timer);
  }, [setBackendStatus]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-slate-800/60 py-4 text-center text-xs text-slate-600">
        EKF 多目标检测与轨迹预测系统 &nbsp;·&nbsp; 基于扩展卡尔曼滤波 · CTRV 运动模型
      </footer>
    </div>
  );
}
