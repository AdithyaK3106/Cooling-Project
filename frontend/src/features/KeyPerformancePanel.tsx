import { useTelemetry } from '../services/telemetryApi';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { useUiStore } from '../stores/uiStore';
import { Leaf, Zap, Activity } from 'lucide-react';

export function KeyPerformancePanel() {
  const { data: telemetry } = useTelemetry();
  const { uiThemeMode } = useUiStore();
  const isTelemetry = uiThemeMode === 'TELEMETRY';

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

  const containerClass = isTelemetry
    ? 'rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl font-sans'
    : 'rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md font-sans';

  const headerTitleClass = isTelemetry
    ? 'text-xs font-bold tracking-wider text-gray-400 uppercase font-mono'
    : 'text-[11px] font-bold tracking-widest text-neutral-400 uppercase font-mono';

  const pueTextClass = isTelemetry
    ? 'text-3xl font-bold font-mono text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.4)]'
    : 'text-2xl font-mono text-neutral-100 font-bold';

  const pueBarClass = isTelemetry
    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-[0_0_8px_#06b6d4]'
    : 'bg-zinc-400';

  return (
    <>
      {/* PUE Card */}
      <div className={containerClass}>
        <div className="flex justify-between items-center mb-2.5">
          <h2 className={headerTitleClass}>Facility PUE</h2>
          <Zap size={15} className={isTelemetry ? 'text-cyan-400' : 'text-neutral-300'} />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <div className={pueTextClass}>{animatedPUE}</div>
            <div className={`text-[10px] font-mono mt-0.5 ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>TARGET: 1.00</div>
          </div>
          <div className={`text-right text-xs font-mono ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>
            <div>EFFICIENCY</div>
            <div className={`font-bold ${isTelemetry ? 'text-cyan-300' : 'text-neutral-200'}`}>{(100 / pue).toFixed(1)}%</div>
          </div>
        </div>
        
        <div className={`mt-3.5 h-1.5 w-full rounded-sm overflow-hidden flex ${isTelemetry ? 'bg-white/10' : 'bg-neutral-900'}`}>
          <div className={`h-full transition-all duration-500 ${pueBarClass}`} style={{ width: `${Math.min(100, Math.max(10, (2.0 - pue) * 100))}%` }} />
        </div>
      </div>

      {/* Cooling Energy Card */}
      <div className={containerClass}>
        <div className="flex justify-between items-center mb-2.5">
          <h2 className={headerTitleClass}>Cooling Power Draw</h2>
          <Activity size={15} className={isTelemetry ? 'text-cyan-400' : 'text-neutral-300'} />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <div className={isTelemetry ? 'text-3xl font-bold font-mono text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.4)]' : 'text-2xl font-mono text-neutral-100 font-bold'}>
              {useAnimatedNumber(totalPower * 0.35 / 1000, 600, (v: number) => v.toFixed(2))}
            </div>
            <div className={`text-[10px] font-mono mt-0.5 ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>MW TOTAL LOAD</div>
          </div>
          <div className={`text-right text-xs font-mono ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>
            <div>COOLING RACKS</div>
            <div className={`font-bold ${isTelemetry ? 'text-cyan-300' : 'text-neutral-200'}`}>{coolingRacks} / {racks.length || 25}</div>
          </div>
        </div>
        
        <div className="mt-3.5 flex items-end justify-between h-3.5 gap-1 opacity-80">
          {racks.slice(0, 15).map((r: any, i: number) => (
            <div 
              key={i} 
              className={`w-full rounded-t-sm transition-all duration-300 ${
                r.risk_score > 0.55 
                  ? 'bg-red-500' 
                  : isTelemetry 
                  ? 'bg-cyan-500 shadow-[0_0_5px_#06b6d4]' 
                  : 'bg-zinc-600'
              }`} 
              style={{ height: `${Math.max(15, r.risk_score * 100)}%` }} 
            />
          ))}
        </div>
      </div>

      {/* Model & Fleet Risk Card */}
      <div className={containerClass}>
        <div className="flex justify-between items-center mb-2.5">
          <h2 className={headerTitleClass}>ML Model & Fleet Risk</h2>
          <Leaf size={15} className={isTelemetry ? 'text-green-400' : 'text-emerald-400'} />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <div className={`text-2xl font-mono font-bold ${isTelemetry ? 'text-green-400 drop-shadow-[0_0_8px_rgba(74,222,128,0.4)]' : 'text-emerald-400'}`}>{animatedAccuracy}%</div>
            <div className={`text-[10px] font-mono mt-0.5 ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>XGBOOST + GNN ACCURACY</div>
          </div>

          <div className="text-right">
            <div className={`text-xl font-mono font-bold ${isTelemetry ? 'text-amber-300 drop-shadow-[0_0_8px_rgba(252,211,77,0.4)]' : 'text-amber-400'}`}>{animatedAvgRisk}%</div>
            <div className={`text-[10px] font-mono mt-0.5 ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>AVG THERMAL RISK</div>
          </div>
        </div>

        <div className={`mt-3.5 flex justify-between text-[11px] font-mono border-t pt-2 ${isTelemetry ? 'text-gray-400 border-white/10' : 'text-neutral-400 border-neutral-800/80'}`}>
          <span>Active Alerts: <strong className="text-amber-400">{modelStats.active_alerts ?? criticalRacks}</strong></span>
          <span>Hot Zones: <strong className="text-red-400">{criticalRacks}</strong></span>
        </div>
      </div>
    </>
  );
}
