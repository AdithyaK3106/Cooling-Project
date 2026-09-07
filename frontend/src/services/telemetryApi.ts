import { useQuery } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import { tickSimulation, getSimulatedTelemetry } from './simulation';

export const TELEMETRY_QUERY_KEY = ['telemetry'];

// Start the simulation loop
setInterval(() => {
  tickSimulation();
}, 200);

export async function getTelemetry(): Promise<any> {
  const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
  if (mode === 'SIMULATED') {
    return getSimulatedTelemetry();
  }
  try {
    return await fetchApi<any>('/telemetry');
  } catch {
    return getSimulatedTelemetry();
  }
}

export function useTelemetry(pollingIntervalMs = 100) {
  return useQuery({
    queryKey: TELEMETRY_QUERY_KEY,
    queryFn: getTelemetry,
    refetchInterval: pollingIntervalMs, // 10Hz telemetry ticks
    retry: 3,
  });
}
