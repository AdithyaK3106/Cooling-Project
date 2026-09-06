import { useTelemetry } from '../services/telemetryApi';

export function StatusStrip() {
  const { data: telemetry } = useTelemetry();
  const racks = telemetry?.racks || [];
  
  const hotZones = racks.filter((r) => r.risk_score > 0.55).length;
  const coolingActiveCount = racks.filter((r) => r.cooling?.status === 'predictive intervention').length;

  return (
    <div className="flex h-10 shrink-0 items-center gap-6 overflow-x-auto border-b border-white/5 bg-[#0B0E14]/80 px-6 backdrop-blur">
      <div className="flex items-center gap-2 whitespace-nowrap">
        <span className="text-[10px] font-bold tracking-wider text-gray-500">SYSTEM STATUS</span>
        <span className="text-xs font-bold text-green-400">Operational</span>
      </div>
      <div className="h-4 w-px bg-white/10" />
      <div className="flex items-center gap-2 whitespace-nowrap">
        <span className="text-[10px] font-bold tracking-wider text-gray-500">THERMAL RISK</span>
        <span className={`text-xs font-bold ${hotZones > 0 ? 'text-orange-400' : 'text-gray-300'}`}>
          {hotZones} racks require attention
        </span>
      </div>
      <div className="h-4 w-px bg-white/10" />
      <div className="flex items-center gap-2 whitespace-nowrap">
        <span className="text-[10px] font-bold tracking-wider text-gray-500">COOLING INTERVENTIONS</span>
        <span className="text-xs font-bold text-cyan-400">{coolingActiveCount} active</span>
      </div>
      <div className="h-4 w-px bg-white/10" />
      <div className="flex items-center gap-2 whitespace-nowrap">
        <span className="text-[10px] font-bold tracking-wider text-gray-500">MODEL INFERENCE</span>
        <span className="text-xs font-bold text-gray-300">Live (GNN + XGB)</span>
      </div>
    </div>
  );
}
