import { useTelemetry } from '../../services/telemetryApi';
import { useUiStore } from '../../stores/uiStore';
import { Activity, Thermometer, Fan, AlertTriangle } from 'lucide-react';

export function RackInspector() {
  const selectedRackId = useUiStore((state) => state.selectedRackId);
  const { data: telemetry } = useTelemetry();

  if (!selectedRackId) return null;

  const rack = telemetry?.racks?.find((r) => r.id === selectedRackId);
  if (!rack) return null;

  return (
    <div className="absolute right-4 top-24 z-10 w-80 rounded-lg border border-thervo-border bg-thervo-panel/90 p-4 text-thervo-text backdrop-blur-md">
      <h2 className="mb-4 text-xl font-mono text-thervo-cool">RACK {rack.id}</h2>
      
      <div className="space-y-4 font-sans">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2"><Activity size={16}/> CPU Util</div>
          <span className="font-mono">{rack.telemetry.cpu_util}%</span>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2"><Thermometer size={16}/> Temp</div>
          <span className="font-mono">{rack.telemetry.cpu_temp}°C</span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2"><Fan size={16}/> Cooling</div>
          <span className="font-mono">{rack.cooling.actual_rpm} RPM</span>
        </div>

        <div className={`mt-4 rounded p-2 ${rack.risk_score > 0.6 ? 'bg-thervo-critical/20 text-thervo-critical' : 'bg-thervo-cool/10 text-thervo-cool'}`}>
          <div className="flex items-center gap-2">
            <AlertTriangle size={16} /> 
            Risk Score: {rack.risk_score.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
