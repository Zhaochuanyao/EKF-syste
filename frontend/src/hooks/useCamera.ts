/**
 * 摄像头管理 Hook
 * 封装 getUserMedia、设备枚举、流控制
 */
import { useState, useRef, useEffect, useCallback } from 'react';

export type CameraStatus =
  | 'idle'
  | 'requesting'
  | 'active'
  | 'error';

export interface CameraState {
  status: CameraStatus;
  error: string | null;
  devices: MediaDeviceInfo[];
  selectedDeviceId: string;
}

export interface UseCameraReturn {
  state: CameraState;
  videoRef: React.RefObject<HTMLVideoElement>;
  startCamera: (deviceId?: string) => Promise<void>;
  stopCamera: () => void;
  selectDevice: (deviceId: string) => void;
}

export function useCamera(): UseCameraReturn {
  const [state, setState] = useState<CameraState>({
    status: 'idle',
    error: null,
    devices: [],
    selectedDeviceId: '',
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  /** 枚举可用视频设备 */
  const enumerateDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = all.filter((d) => d.kind === 'videoinput');
      setState((s) => ({
        ...s,
        devices: videoDevices,
        selectedDeviceId: s.selectedDeviceId || videoDevices[0]?.deviceId || '',
      }));
    } catch {
      // 枚举失败不阻断流程
    }
  }, []);

  /** 打开摄像头 */
  const startCamera = useCallback(
    async (deviceId?: string) => {
      // 先停止旧流
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      setState((s) => ({ ...s, status: 'requesting', error: null }));

      if (!navigator.mediaDevices?.getUserMedia) {
        setState((s) => ({ ...s, status: 'error', error: '浏览器不支持摄像头访问，请使用 HTTPS 或 localhost' }));
        return;
      }

      try {
        const targetDevice = deviceId || state.selectedDeviceId;
        const constraints: MediaStreamConstraints = {
          video: targetDevice
            ? { deviceId: { exact: targetDevice }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {/* autoplay policy */});
        }

        setState((s) => ({ ...s, status: 'active', error: null }));
        await enumerateDevices();
      } catch (err) {
        const msg =
          err instanceof DOMException
            ? err.name === 'NotAllowedError'
              ? '摄像头权限被拒绝，请在浏览器设置中允许访问'
              : err.name === 'NotFoundError'
              ? '未找到摄像头设备'
              : err.message
            : String(err);
        setState((s) => ({ ...s, status: 'error', error: msg }));
      }
    },
    [state.selectedDeviceId, enumerateDevices],
  );

  /** 关闭摄像头 */
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setState((s) => ({ ...s, status: 'idle', error: null }));
  }, []);

  const selectDevice = useCallback((deviceId: string) => {
    setState((s) => ({ ...s, selectedDeviceId: deviceId }));
  }, []);

  // 组件卸载时释放摄像头
  useEffect(() => {
    enumerateDevices();
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [enumerateDevices]);

  return { state, videoRef, startCamera, stopCamera, selectDevice };
}
