import { useEffect } from 'react';
import { useTelemetry } from './telemetryApi';
import { useAnalyticsStore } from '../stores/analyticsStore';

export function TelemetryRecorder() {
  const { data } = useTelemetry();
  const appendTelemetry = useAnalyticsStore(state => state.appendTelemetry);

  useEffect(() => {
    if (data) {
      appendTelemetry(data);
    }
  }, [data, appendTelemetry]);

  return null;
}
