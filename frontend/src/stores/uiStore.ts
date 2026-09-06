import { create } from 'zustand';

interface UiState {
  selectedRackId: string | null;
  setSelectedRackId: (id: string | null) => void;
  // We can add more UI state here like panel visibility, current view mode, etc.
}

export const useUiStore = create<UiState>((set) => ({
  selectedRackId: null,
  setSelectedRackId: (id) => set({ selectedRackId: id }),
}));
