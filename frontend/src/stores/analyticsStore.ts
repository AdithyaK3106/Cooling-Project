import { create } from 'zustand';
import type { ThervoTelemetry } from '../types/telemetry';

interface HistoryPoint {
  time: string;
  avgCpu: number;
  avgTemp: number;
  maxRisk: number;
}

interface AnalyticsState {
  history: HistoryPoint[];
  appendTelemetry: (data: ThervoTelemetry) => void;
}

export const useAnalyticsStore = create<AnalyticsState>((set) => ({
  history: [],
  appendTelemetry: (data) => set((state) => {
    // Only keep last 60 points (1 minute at 1Hz)
    const newHistory = [...state.history];
    if (newHistory.length >= 60) newHistory.shift();

    let totalCpu = 0, totalTemp = 0, maxRisk = 0;
    const racks = data.racks || [];
    
    if (racks.length === 0) return { history: state.history };

    racks.forEach(r => {
      totalCpu += r.telemetry.cpu_util;
      totalTemp += r.telemetry.cpu_temp;
      if (r.risk_score > maxRisk) maxRisk = r.risk_score;
    });

    newHistory.push({
      time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      avgCpu: totalCpu / racks.length,
      avgTemp: totalTemp / racks.length,
      maxRisk: maxRisk
    });

    return { history: newHistory };
  })
}));
