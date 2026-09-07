import { useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import { TELEMETRY_QUERY_KEY } from './telemetryApi';
import { setSimulationParams, injectRandomSpike, injectRackSpike, toggleRackOverride, resetSimulation } from './simulation';

export function useUpdateSimulationControls() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: { load: number; noise: number }) => {
      const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
      if (mode === 'SIMULATED') {
        setSimulationParams(params.load, params.noise);
        return { success: true };
      }
      return fetchApi('/control/simulation', { method: 'POST', body: JSON.stringify(params) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}

export function useInjectSpike() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (rackId?: string) => {
      const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
      if (mode === 'SIMULATED') {
        if (rackId) injectRackSpike(rackId);
        else injectRandomSpike();
        return { success: true };
      }
      return fetchApi('/control/spike', { method: 'POST', body: JSON.stringify({ rackId }) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}

export function useToggleRackOverride() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (rackId: string) => {
      const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
      if (mode === 'SIMULATED') {
        toggleRackOverride(rackId);
        return { success: true };
      }
      return fetchApi('/control/override', { method: 'POST', body: JSON.stringify({ rackId }) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}

export function useResetSimulation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
      if (mode === 'SIMULATED') {
        resetSimulation();
        return { success: true };
      }
      return fetchApi('/control/reset', { method: 'POST', body: JSON.stringify({}) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}

import { setRackCooling } from './simulation';

export function useSetRackCooling() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ rackId, status }: { rackId: string; status: 'predictive intervention' | 'normal' }) => {
      setRackCooling(rackId, status === 'predictive intervention');
      try {
        await fetchApi('/control/cooling', { method: 'POST', body: JSON.stringify({ rackId, status }) });
      } catch {
        // Fallback for simulation / offline mode
      }
      return { success: true };
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}
