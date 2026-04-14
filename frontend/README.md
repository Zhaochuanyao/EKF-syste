# EKF 多目标跟踪系统 - 前端

基于 React + TypeScript + Vite + Tailwind CSS 的现代化 Web 界面，用于演示「基于扩展卡尔曼滤波的目标检测与运动轨迹预测系统」。

## 功能

| 页面 | 功能 |
|------|------|
| 总控台 | 系统介绍、后端连接状态、功能入口 |
| 视频上传预测 | 拖拽/点击上传视频 → 异步处理 → 播放结果视频 + 下载 |
| 摄像头实时预测 | 调用浏览器摄像头 → 逐帧推理 → canvas 叠加绘制检测框/轨迹/预测点 |

## 技术栈

- **React 18** + **TypeScript**
- **Vite 5** 构建工具
- **Tailwind CSS 3** 样式
- **Zustand** 状态管理
- **Axios** HTTP 请求
- **React Router 6** 路由
- **Lucide React** 图标

## 快速启动

### 前提条件

1. Node.js 18+（推荐使用 nvm 或 fnm 管理）
2. 后端已启动（见下方）

### 安装依赖

```bash
cd frontend
npm install
```

### 启动开发服务器

```bash
npm run dev
```

浏览器访问 http://localhost:5173

### 构建生产版本

```bash
npm run build
```

---

## 与后端联调

### 1. 启动后端

在项目根目录：

```bash
# 安装后端依赖
pip install -r requirements.txt

# 启动 FastAPI 服务
uvicorn src.ekf_mot.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

后端 API 文档：http://localhost:8000/docs

### 2. 配置前端 API 地址

前端读取 `.env` 文件中的 `VITE_API_BASE_URL`，默认为 `http://localhost:8000`。

如需修改，编辑 `frontend/.env`：

```env
VITE_API_BASE_URL=http://你的后端地址:8000
```

---

## 目录结构

```
frontend/
├── src/
│   ├── pages/
│   │   ├── DashboardPage.tsx      # 总控台
│   │   ├── UploadPredictPage.tsx  # 视频上传预测
│   │   └── CameraPredictPage.tsx  # 摄像头实时预测
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Navbar.tsx         # 顶部导航栏
│   │   │   └── Layout.tsx         # 布局容器（含后端健康检测）
│   │   └── common/
│   │       ├── StatusBadge.tsx    # 状态徽章
│   │       └── LoadingSpinner.tsx # 加载指示器
│   ├── services/
│   │   ├── api.ts                 # API 封装（axios）
│   │   └── types.ts               # TypeScript 类型定义
│   ├── hooks/
│   │   └── useCamera.ts           # 摄像头管理 Hook
│   ├── store/
│   │   └── appStore.ts            # 全局状态（Zustand）
│   └── styles/
│       └── globals.css            # Tailwind + 全局样式
```

---

## 后端接口说明

前端使用以下后端接口：

| 方法 | 路径 | 用途 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| POST | `/predict/frame` | 单帧推理（摄像头模式） |
| POST | `/predict/video/start` | 启动异步视频处理任务 |
| GET  | `/predict/video/status/{task_id}` | 查询任务进度 |
| GET  | `/outputs/{filename}` | 获取输出视频文件 |
| POST | `/reset` | 重置跟踪器 |

---

## 测试流程

### 视频上传预测

1. 确认后端在线（Navbar 右上角绿色「后端在线」）
2. 进入「视频上传预测」页面
3. 拖拽或点击上传 `.mp4` / `.avi` / `.mov` 视频
4. 配置参数（默认即可）
5. 点击「开始处理」
6. 等待上传进度 → 处理进度（2 秒轮询一次）
7. 完成后查看结果视频，点击「下载视频」

### 摄像头实时预测

1. 进入「摄像头实时预测」页面
2. 点击「打开摄像头」，允许浏览器权限申请
3. 点击「开始预测」
4. 画面上将叠加检测框（彩色）、轨迹 ID、未来预测点（青色虚线）
5. 右侧实时显示轨迹数量、推理延迟等统计信息
6. 可随时「暂停」/ 继续 / 「停止」预测

---

## 常见问题

**Q: 后端显示离线？**  
A: 确认 `uvicorn` 已启动，且 `.env` 中地址正确。

**Q: 摄像头权限被拒绝？**  
A: 在浏览器地址栏左侧点击锁图标，手动允许摄像头权限，然后刷新页面。

**Q: 视频处理很慢？**  
A: 后端默认使用 CPU，首次加载 YOLOv8 模型需要 15~30 秒。
可在 `configs/base.yaml` 中设置 `device: cuda` 使用 GPU 加速。
