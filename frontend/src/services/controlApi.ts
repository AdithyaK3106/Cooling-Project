import { useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import { TELEMETRY_QUERY_KEY } from './telemetryApi';

export function useUpdateSimulationControls() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: { load: number; noise: number }) => {
      return fetchApi('/control/simulation', { method: 'POST', body: JSON.stringify(params) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}

export function useInjectSpike() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      return fetchApi('/control/spike', { method: 'POST', body: JSON.stringify({}) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY }),
  });
}
