import { useQuery } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import type { ThervoTelemetry } from '../types/telemetry';

export const TELEMETRY_QUERY_KEY = ['telemetry'];

export async function getTelemetry(): Promise<ThervoTelemetry> {
  return fetchApi<ThervoTelemetry>('/telemetry');
}

export function useTelemetry(pollingIntervalMs = 100) {
  return useQuery({
    queryKey: TELEMETRY_QUERY_KEY,
    queryFn: getTelemetry,
    refetchInterval: pollingIntervalMs, // 10Hz telemetry ticks
    retry: 3,
  });
}
