import { useTelemetry } from '../../services/telemetryApi';
import { Terminal } from 'lucide-react';

export function EventStream() {
  const { data: telemetry } = useTelemetry();

  if (!telemetry?.events?.length) return null;

  return (
    <div className="absolute right-4 bottom-4 z-10 w-96 rounded-lg border border-thervo-border bg-thervo-panel/90 p-4 text-thervo-text backdrop-blur-md">
      <div className="mb-2 flex items-center gap-2 text-sm font-bold text-thervo-cool">
        <Terminal size={14} /> SYSTEM EVENTS
      </div>
      <div className="flex h-32 flex-col gap-1 overflow-y-auto font-mono text-xs">
        {telemetry.events.map((event: any, i: number) => (
          <div key={i} className="flex gap-2">
            <span className="text-thervo-amber">[{event.time}]</span>
            <span className="text-gray-300">{event.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
