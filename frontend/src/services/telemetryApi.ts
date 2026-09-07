import { useQuery } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import { tickSimulation, getSimulatedTelemetry } from './simulation';

export const TELEMETRY_QUERY_KEY = ['telemetry'];

// Start the simulation loop
setInterval(() => {
  if (localStorage.getItem('thervo_mode') === 'SIMULATED') {
    tickSimulation();
  }
}, 250);

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
