/**
 * 总控台 / 首页
 */
import { Link } from 'react-router-dom';
import {
  Activity,
  Video,
  Camera,
  Eye,
  Cpu,
  TrendingUp,
  BarChart3,
  ArrowRight,
  CheckCircle2,
  XCircle,
  Loader2,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const FEATURES = [
  {
    icon: Eye,
    title: '车辆目标检测',
    desc: 'YOLOv8 实时检测，专项过滤车辆类别（轿车/卡车/巴士/摩托），高精度低延迟',
    color: 'text-blue-400',
    bg: 'bg-blue-400/10',
  },
  {
    icon: Activity,
    title: '车辆多目标跟踪',
    desc: '四阶段匈牙利关联 + 遮挡恢复机制，针对道路场景优化轨迹碎片化抑制',
    color: 'text-cyan-400',
    bg: 'bg-cyan-400/10',
  },
  {
    icon: TrendingUp,
    title: '车辆轨迹预测',
    desc: 'CTRV 运动模型 + 扩展卡尔曼滤波，预测车辆未来 1/5/10 帧行驶轨迹',
    color: 'text-violet-400',
    bg: 'bg-violet-400/10',
  },
  {
    icon: BarChart3,
    title: '交通场景可视化',
    desc: '实时叠加车辆检测框、跟踪 ID、历史轨迹线与短时预测路径',
    color: 'text-emerald-400',
    bg: 'bg-emerald-400/10',
  },
];

const TECH_STACK = [
  'Extended Kalman Filter',
  'CTRV Motion Model',
  'YOLOv8n Detector',
  'Hungarian Algorithm',
  '3-Stage Association',
  'Bootstrap Kinematics',
  'Adaptive Process Noise',
  'Prediction Gate',
  'Fixed-Lag Smoothing',
];

function BackendStatusBanner() {
  const status = useAppStore((s) => s.backendStatus);

  if (status === 'online') {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
        <CheckCircle2 className="w-4 h-4" />
        后端服务已连接 · 准备就绪
      </div>
    );
  }
  if (status === 'offline') {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
        <XCircle className="w-4 h-4" />
        后端服务未连接 · 请启动 uvicorn
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800 border border-slate-700 text-slate-400 text-sm">
      <Loader2 className="w-4 h-4 animate-spin" />
      正在检测后端连接...
    </div>
  );
}

export default function DashboardPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12 space-y-16 animate-fade-in">
      {/* Hero */}
      <section className="text-center space-y-6">
        <div className="flex justify-center">
          <BackendStatusBanner />
        </div>

        <div>
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-300 bg-clip-text text-transparent leading-tight pb-1">
            EKF 车辆多目标跟踪系统
          </h1>
          <p className="mt-4 text-slate-400 text-lg sm:text-xl max-w-2xl mx-auto">
            面向交通监控的车辆多目标跟踪与短时轨迹预测平台
            <br />
            <span className="text-slate-500 text-base">
              毕业设计演示系统 · CTRV 运动模型 · 四阶段关联 · 预测质量门控
            </span>
          </p>
        </div>

        {/* 入口卡片 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 max-w-2xl mx-auto pt-2">
          <Link
            to="/upload"
            className="group card p-6 text-left hover:border-blue-500/50 hover:bg-slate-900/80 transition-all duration-200 hover:shadow-lg hover:shadow-blue-500/5"
          >
            <div className="flex items-start justify-between">
              <div className="w-11 h-11 rounded-xl bg-blue-500/15 flex items-center justify-center mb-4">
                <Video className="w-5 h-5 text-blue-400" />
              </div>
              <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-blue-400 group-hover:translate-x-1 transition-all" />
            </div>
            <h3 className="text-white font-semibold mb-1.5">视频上传预测</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              上传交通监控或行车记录视频，批量跟踪车辆，生成带标注的结果视频并下载
            </p>
            <div className="mt-4 flex items-center gap-1.5 text-blue-400 text-xs font-medium">
              <span>支持 MP4 / AVI / MOV · 车辆优先</span>
            </div>
          </Link>

          <Link
            to="/camera"
            className="group card p-6 text-left hover:border-cyan-500/50 hover:bg-slate-900/80 transition-all duration-200 hover:shadow-lg hover:shadow-cyan-500/5"
          >
            <div className="flex items-start justify-between">
              <div className="w-11 h-11 rounded-xl bg-cyan-500/15 flex items-center justify-center mb-4">
                <Camera className="w-5 h-5 text-cyan-400" />
              </div>
              <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" />
            </div>
            <h3 className="text-white font-semibold mb-1.5">摄像头实时预测</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              调用摄像头进行实时车辆跟踪演示，画面上叠加检测框、轨迹与行驶预测点
            </p>
            <div className="mt-4 flex items-center gap-1.5 text-cyan-400 text-xs font-medium">
              <span>需要摄像头权限 · 车辆优先</span>
            </div>
          </Link>
        </div>
      </section>

      {/* 系统能力 */}
      <section>
        <h2 className="text-xl font-semibold text-slate-200 mb-6 text-center">系统核心能力（车辆场景优化）</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map(({ icon: Icon, title, desc, color, bg }) => (
            <div key={title} className="card p-5 space-y-3">
              <div className={`w-9 h-9 rounded-lg ${bg} flex items-center justify-center`}>
                <Icon className={`w-4.5 h-4.5 ${color}`} style={{ width: 18, height: 18 }} />
              </div>
              <h3 className="font-semibold text-slate-200">{title}</h3>
              <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 技术标签 */}
      <section>
        <h2 className="text-xl font-semibold text-slate-200 mb-5 text-center">核心技术</h2>
        <div className="flex flex-wrap justify-center gap-2">
          {TECH_STACK.map((tech) => (
            <span
              key={tech}
              className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 text-sm font-mono"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* 使用说明 */}
      <section className="card p-6 sm:p-8">
        <h2 className="text-lg font-semibold text-slate-200 mb-5 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-blue-400" />
          快速开始
        </h2>
        <ol className="space-y-3 text-sm text-slate-400">
          <li className="flex gap-3">
            <span className="shrink-0 w-6 h-6 rounded-full bg-blue-600/20 border border-blue-600/30 text-blue-400 text-xs flex items-center justify-center font-bold">1</span>
            <span>确认后端已启动：<code className="bg-slate-800 px-2 py-0.5 rounded text-slate-300 font-mono text-xs">uvicorn src.ekf_mot.serving.api:app --host 0.0.0.0 --port 8000</code></span>
          </li>
          <li className="flex gap-3">
            <span className="shrink-0 w-6 h-6 rounded-full bg-blue-600/20 border border-blue-600/30 text-blue-400 text-xs flex items-center justify-center font-bold">2</span>
            <span>确认右上角后端状态显示「在线」（绿色）</span>
          </li>
          <li className="flex gap-3">
            <span className="shrink-0 w-6 h-6 rounded-full bg-blue-600/20 border border-blue-600/30 text-blue-400 text-xs flex items-center justify-center font-bold">3</span>
            <span>选择「视频上传预测」处理本地视频，或「摄像头实时预测」进行实时演示</span>
          </li>
        </ol>
      </section>
    </div>
  );
}
