import { useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchApi } from './apiClient';
import type { ScenarioParams } from '../types/telemetry';
import { TELEMETRY_QUERY_KEY } from './telemetryApi';

export async function updateSimulationControls(params: ScenarioParams): Promise<void> {
  return fetchApi('/simulation/controls', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export function useUpdateSimulationControls() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: updateSimulationControls,
    onSuccess: () => {
      // Invalidate and refetch telemetry immediately after a control update
      queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY });
    },
  });
}
