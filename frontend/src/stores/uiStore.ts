import { create } from 'zustand';

interface UiState {
  selectedRackId: string | null;
  hoveredRackId: string | null;
  setSelectedRackId: (id: string | null) => void;
  setHoveredRackId: (id: string | null) => void;
  // We can add more UI state here like panel visibility, current view mode, etc.
}

export const useUiStore = create<UiState>((set) => ({
  selectedRackId: null,
  hoveredRackId: null,
  setSelectedRackId: (id) => set({ selectedRackId: id }),
  setHoveredRackId: (id) => set({ hoveredRackId: id }),
}));
