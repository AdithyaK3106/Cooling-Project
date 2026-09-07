import { useTelemetry } from '../services/telemetryApi';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { Leaf, Zap, Activity } from 'lucide-react';

export function KeyPerformancePanel() {
  const { data: telemetry } = useTelemetry();

  const racks = telemetry?.racks || [];
  const modelStats = telemetry?.model_stats || {};

  // Calculate live dynamic metrics across all racks
  const avgTemp = racks.length > 0 
    ? racks.reduce((acc: number, r: any) => acc + (r.telemetry?.cpu_temp || 45), 0) / racks.length 
    : 45;
    
  const avgRisk = racks.length > 0 
    ? (racks.reduce((acc: number, r: any) => acc + (r.risk_score || 0), 0) / racks.length) * 100 
    : 32;

  const totalPower = racks.length > 0
    ? racks.reduce((acc: number, r: any) => acc + (r.telemetry?.power_draw || 250), 0)
    : 1200;

  const coolingRacks = racks.filter((r: any) => r.cooling?.status === 'predictive intervention' || r.cooling?.override).length;
  const criticalRacks = racks.filter((r: any) => r.risk_score > 0.55).length;
  
  // Calculate PUE (Power Usage Effectiveness) dynamically
  const pue = 1.0 + (avgTemp / 100) * 0.5;
  const animatedPUE = useAnimatedNumber(pue, 600, (v: number) => v.toFixed(2));

  const animatedAccuracy = useAnimatedNumber(modelStats.accuracy || 96.4, 600, (v: number) => v.toFixed(1));
  const animatedAvgRisk = useAnimatedNumber(avgRisk, 600, (v: number) => v.toFixed(1));

  return (
    <>
      {/* PUE Card */}
      <div className="rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md font-sans">
        <div className="flex justify-between items-center mb-2.5">
          <h2 className="text-[11px] font-bold tracking-widest text-neutral-400 uppercase font-mono">Facility PUE</h2>
          <Zap size={15} className="text-neutral-300" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <div className="text-2xl font-mono text-neutral-100 font-bold">{animatedPUE}</div>
            <div className="text-[10px] text-neutral-400 font-mono mt-0.5">TARGET: 1.00</div>
          </div>
          <div className="text-right text-xs font-mono text-neutral-400">
            <div>EFFICIENCY</div>
            <div className="text-neutral-200 font-bold">{(100 / pue).toFixed(1)}%</div>
          </div>
        </div>
        
        <div className="mt-3.5 h-1.5 w-full rounded-sm bg-neutral-900 overflow-hidden flex">
          <div className="h-full bg-zinc-400 transition-all duration-500" style={{ width: `${Math.min(100, Math.max(10, (2.0 - pue) * 100))}%` }} />
        </div>
      </div>

      {/* Cooling Energy Card */}
      <div className="rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md font-sans">
        <div className="flex justify-between items-center mb-2.5">
          <h2 className="text-[11px] font-bold tracking-widest text-neutral-400 uppercase font-mono">Cooling Power Draw</h2>
          <Activity size={15} className="text-neutral-300" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <div className="text-2xl font-mono text-neutral-100 font-bold">
              {useAnimatedNumber(totalPower * 0.35 / 1000, 600, (v: number) => v.toFixed(2))}
            </div>
            <div className="text-[10px] text-neutral-400 font-mono mt-0.5">MW TOTAL LOAD</div>
          </div>
          <div className="text-right text-xs font-mono text-neutral-400">
            <div>COOLING RACKS</div>
            <div className="text-neutral-200 font-bold">{coolingRacks} / {racks.length || 25}</div>
          </div>
        </div>
        
        <div className="mt-3.5 flex items-end justify-between h-3.5 gap-1 opacity-80">
          {racks.slice(0, 15).map((r: any, i: number) => (
            <div 
              key={i} 
              className={`w-full rounded-t-sm transition-all duration-300 ${r.risk_score > 0.55 ? 'bg-red-800' : 'bg-zinc-600'}`} 
              style={{ height: `${Math.max(15, r.risk_score * 100)}%` }} 
            />
          ))}
        </div>
      </div>

      {/* Model & Fleet Risk Card */}
      <div className="rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md font-sans">
        <div className="flex justify-between items-center mb-2.5">
          <h2 className="text-[11px] font-bold tracking-widest text-neutral-400 uppercase font-mono">ML Model & Fleet Risk</h2>
          <Leaf size={15} className="text-emerald-400" />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <div className="text-2xl font-mono text-emerald-400 font-bold">{animatedAccuracy}%</div>
            <div className="text-[10px] text-neutral-400 font-mono mt-0.5">XGBOOST + GNN ACCURACY</div>
          </div>

          <div className="text-right">
            <div className="text-xl font-mono text-amber-400 font-bold">{animatedAvgRisk}%</div>
            <div className="text-[10px] text-neutral-400 font-mono mt-0.5">AVG THERMAL RISK</div>
          </div>
        </div>

        <div className="mt-3.5 flex justify-between text-[11px] font-mono text-neutral-400 border-t border-neutral-800/80 pt-2">
          <span>Active Alerts: <strong className="text-amber-400">{modelStats.active_alerts ?? criticalRacks}</strong></span>
          <span>Hot Zones: <strong className="text-red-400">{criticalRacks}</strong></span>
        </div>
      </div>
    </>
  );
}
