import { create } from 'zustand';

export interface UiState {
  selectedRackId: string | null;
  hoveredRackId: string | null;
  activeLayer: 'THERMAL' | 'AIRFLOW' | 'RISK';
  perspective: '3D' | 'TOP' | 'SIDE';
  uiThemeMode: 'INDUSTRIAL' | 'TELEMETRY';
  setSelectedRackId: (id: string | null) => void;
  setHoveredRackId: (id: string | null) => void;
  setActiveLayer: (layer: 'THERMAL' | 'AIRFLOW' | 'RISK') => void;
  setPerspective: (perspective: '3D' | 'TOP' | 'SIDE') => void;
  setUiThemeMode: (mode: 'INDUSTRIAL' | 'TELEMETRY') => void;
}

const initialTheme = (localStorage.getItem('ui_theme_mode') as 'INDUSTRIAL' | 'TELEMETRY') || 'INDUSTRIAL';

export const useUiStore = create<UiState>((set) => ({
  selectedRackId: null,
  hoveredRackId: null,
  activeLayer: 'THERMAL',
  perspective: '3D',
  uiThemeMode: initialTheme,
  setSelectedRackId: (id) => set({ selectedRackId: id }),
  setHoveredRackId: (id) => set({ hoveredRackId: id }),
  setActiveLayer: (layer) => set({ activeLayer: layer }),
  setPerspective: (perspective) => set({ perspective }),
  setUiThemeMode: (mode) => {
    localStorage.setItem('ui_theme_mode', mode);
    set({ uiThemeMode: mode });
  },
}));
