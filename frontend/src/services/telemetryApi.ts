import { useQuery } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import { tickSimulation, getSimulatedTelemetry, getSimulationConfig, subscribeSimulationConfig } from './simulation';

export const TELEMETRY_QUERY_KEY = ['telemetry'];

let simulationTimer: any = null;

function syncSimulationTimer() {
  if (simulationTimer) clearInterval(simulationTimer);
  const config = getSimulationConfig();
  simulationTimer = setInterval(() => {
    if (localStorage.getItem('thervo_mode') === 'SIMULATED') {
      tickSimulation();
    }
  }, config.simSpeedMs);
}

syncSimulationTimer();
subscribeSimulationConfig(syncSimulationTimer);

export async function getTelemetry(): Promise<any> {
  const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
  if (mode === 'SIMULATED') {
    return getSimulatedTelemetry();
  }
  return fetchApi<any>('/telemetry');
}

export function useTelemetry(pollingIntervalMs = 100) {
  return useQuery({
    queryKey: TELEMETRY_QUERY_KEY,
    queryFn: getTelemetry,
    refetchInterval: pollingIntervalMs, // 10Hz telemetry ticks
    retry: 3,
  });
}
