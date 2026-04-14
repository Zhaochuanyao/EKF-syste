/**
 * 全局应用状态（Zustand）
 */
import { create } from 'zustand';

type BackendStatus = 'unknown' | 'online' | 'offline';

interface AppState {
  backendStatus: BackendStatus;
  setBackendStatus: (status: BackendStatus) => void;
}

export const useAppStore = create<AppState>((set) => ({
  backendStatus: 'unknown',
  setBackendStatus: (status) => set({ backendStatus: status }),
}));
