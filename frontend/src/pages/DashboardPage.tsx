import { Link } from 'react-router-dom';
import {
  Activity, Video, Camera, Eye, Cpu, TrendingUp, BarChart3,
  ArrowRight, CheckCircle2, XCircle, Loader2, Lightbulb,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const FEATURES = [
  { icon: Eye,       title: '车辆目标检测',   desc: 'YOLOv8 实时检测，专项过滤车辆类别，高精度低延迟',           color: 'text-blue-400',   bg: 'bg-blue-400/10' },
  { icon: Activity,  title: '多目标轨迹跟踪', desc: '四阶段匈牙利关联 + 遮挡恢复，抑制道路场景轨迹碎片化',     color: 'text-cyan-400',   bg: 'bg-cyan-400/10' },
  { icon: TrendingUp,title: 'EKF 轨迹预测',   desc: 'CTRV 运动模型 + 扩展卡尔曼滤波，预测未来 1/5/10 帧轨迹', color: 'text-violet-400', bg: 'bg-violet-400/10' },
  { icon: BarChart3, title: '自适应噪声调节', desc: '基于新息统计动态调整 R/Q，提升转弯与异常场景鲁棒性',       color: 'text-emerald-400',bg: 'bg-emerald-400/10' },
];

const INNOVATION_POINTS = [
  '观测异常时，动态调大 R，降低异常检测框对滤波器的影响；',
  '目标机动时，动态调大 Q，提高转弯、变速场景的轨迹适应性；',
  '极端异常时，引入裁剪或跳过更新机制，减少目标身份切换。',
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
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-12 space-y-14 animate-fade-in">

      {/* Hero */}
      <section className="text-center space-y-6">
        <div className="flex justify-center">
          <BackendStatusBanner />
        </div>
        <div>
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-300 bg-clip-text text-transparent leading-tight pb-1">
            EKF 车辆多目标跟踪与轨迹预测系统
          </h1>
          <p className="mt-4 text-slate-400 text-lg max-w-2xl mx-auto">
            面向交通监控场景的视频目标检测、轨迹跟踪与短时运动预测平台
          </p>
          <p className="mt-2 text-slate-600 text-sm">
            毕业设计演示系统 · CTRV 运动模型 · 四阶段关联 · 预测质量门控 · R/Q 自适应噪声
          </p>
        </div>

        {/* 入口卡片 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 max-w-2xl mx-auto pt-2">
          <Link
            to="/upload"
            className="group card p-6 text-left hover:border-blue-500/50 transition-all duration-200 hover:shadow-lg hover:shadow-blue-500/5"
          >
            <div className="flex items-start justify-between">
              <div className="w-11 h-11 rounded-xl bg-blue-500/15 flex items-center justify-center mb-4">
                <Video className="w-5 h-5 text-blue-400" />
              </div>
              <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-blue-400 group-hover:translate-x-1 transition-all" />
            </div>
            <h3 className="text-white font-semibold mb-1.5">视频上传预测</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              上传交通监控视频，批量跟踪车辆，生成带标注的结果视频
            </p>
            <div className="mt-4 text-blue-400 text-xs font-medium">进入预测 →</div>
          </Link>

          <Link
            to="/camera"
            className="group card p-6 text-left hover:border-cyan-500/50 transition-all duration-200 hover:shadow-lg hover:shadow-cyan-500/5"
          >
            <div className="flex items-start justify-between">
              <div className="w-11 h-11 rounded-xl bg-cyan-500/15 flex items-center justify-center mb-4">
                <Camera className="w-5 h-5 text-cyan-400" />
              </div>
              <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" />
            </div>
            <h3 className="text-white font-semibold mb-1.5">摄像头实时预测</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              调用摄像头实时跟踪车辆，叠加检测框、轨迹与预测点
            </p>
            <div className="mt-4 text-cyan-400 text-xs font-medium">进入预测 →</div>
          </Link>
        </div>
      </section>

      {/* 系统核心能力 */}
      <section>
        <h2 className="section-title">系统核心能力</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map(({ icon: Icon, title, desc, color, bg }) => (
            <div key={title} className="card p-5 space-y-3">
              <div className={`w-9 h-9 rounded-lg ${bg} flex items-center justify-center`}>
                <Icon className={`w-4.5 h-4.5 ${color}`} style={{ width: 18, height: 18 }} />
              </div>
              <h3 className="font-semibold text-slate-200 text-sm">{title}</h3>
              <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 创新点 */}
      <section>
        <div className="card p-6 sm:p-8 border-blue-500/20">
          <div className="flex items-start gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg bg-blue-500/15 flex items-center justify-center shrink-0">
              <Lightbulb className="w-4.5 h-4.5 text-blue-400" style={{ width: 18, height: 18 }} />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-200">
                创新点：基于新息统计的 R/Q 自适应噪声调节
              </h2>
              <p className="text-slate-500 text-sm mt-0.5">
                在标准 EKF 基础上引入动态噪声调节机制，提升复杂交通场景下的跟踪鲁棒性
              </p>
            </div>
          </div>
          <ul className="space-y-3">
            {INNOVATION_POINTS.map((point, i) => (
              <li key={i} className="flex items-start gap-3 text-sm text-slate-400">
                <span className="shrink-0 w-5 h-5 rounded-full bg-blue-600/20 border border-blue-600/30 text-blue-400 text-xs flex items-center justify-center font-bold mt-0.5">
                  {i + 1}
                </span>
                {point}
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* 快速开始 */}
      <section className="card p-6 sm:p-8">
        <h2 className="text-base font-semibold text-slate-200 mb-5 flex items-center gap-2">
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
            <span>确认右上角后端状态显示绿色圆点「后端在线」</span>
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
