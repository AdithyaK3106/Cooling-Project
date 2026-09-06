import { create } from 'zustand';

interface UiState {
  selectedRackId: string | null;
  hoveredRackId: string | null;
  activeLayer: 'THERMAL' | 'AIRFLOW' | 'RISK';
  perspective: '3D' | 'TOP' | 'SIDE';
  setSelectedRackId: (id: string | null) => void;
  setHoveredRackId: (id: string | null) => void;
  setActiveLayer: (layer: 'THERMAL' | 'AIRFLOW' | 'RISK') => void;
  setPerspective: (perspective: '3D' | 'TOP' | 'SIDE') => void;
}

export const useUiStore = create<UiState>((set) => ({
  selectedRackId: null,
  hoveredRackId: null,
  activeLayer: 'THERMAL',
  perspective: '3D',
  setSelectedRackId: (id) => set({ selectedRackId: id }),
  setHoveredRackId: (id) => set({ hoveredRackId: id }),
  setActiveLayer: (layer) => set({ activeLayer: layer }),
  setPerspective: (perspective) => set({ perspective }),
}));
