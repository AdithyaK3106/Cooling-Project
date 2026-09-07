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
      <div className="rounded-xl border border-white/10 bg-[#0B0E14]/90 p-5 backdrop-blur-xl shadow-2xl transition-all hover:border-cyan-500/30">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xs font-bold tracking-widest text-gray-400 uppercase">Facility PUE</h2>
          <Zap size={16} className="text-cyan-400" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <div className="text-3xl font-mono text-cyan-400 font-bold">{animatedPUE}</div>
            <div className="text-[10px] text-gray-500 font-mono mt-1">TARGET: 1.00</div>
          </div>
          <div className="text-right text-xs font-mono text-gray-400">
            <div>EFFICIENCY</div>
            <div className="text-cyan-400 font-bold">{(100 / pue).toFixed(1)}%</div>
          </div>
        </div>
        
        <div className="mt-4 h-1.5 w-full rounded-full bg-gray-800 overflow-hidden flex">
          <div className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-500" style={{ width: `${Math.min(100, Math.max(10, (2.0 - pue) * 100))}%` }} />
        </div>
      </div>

      {/* Cooling Energy Card */}
      <div className="rounded-xl border border-white/10 bg-[#0B0E14]/90 p-5 backdrop-blur-xl shadow-2xl transition-all hover:border-blue-500/30">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xs font-bold tracking-widest text-gray-400 uppercase">Cooling Power Draw</h2>
          <Activity size={16} className="text-blue-400" />
        </div>
        <div className="flex items-end justify-between">
          <div>
            <div className="text-3xl font-mono text-blue-400 font-bold">
              {useAnimatedNumber(totalPower * 0.35 / 1000, 600, (v: number) => v.toFixed(2))}
            </div>
            <div className="text-[10px] text-gray-500 font-mono mt-1">MW TOTAL LOAD</div>
          </div>
          <div className="text-right text-xs font-mono text-gray-400">
            <div>COOLING RACKS</div>
            <div className="text-blue-400 font-bold">{coolingRacks} / {racks.length || 25}</div>
          </div>
        </div>
        
        <div className="mt-4 flex items-end justify-between h-4 gap-1 opacity-80">
          {racks.slice(0, 15).map((r: any, i: number) => (
            <div 
              key={i} 
              className={`w-full rounded-t-sm transition-all duration-300 ${r.risk_score > 0.55 ? 'bg-red-500' : 'bg-blue-500'}`} 
              style={{ height: `${Math.max(15, r.risk_score * 100)}%` }} 
            />
          ))}
        </div>
      </div>

      {/* Model & Fleet Risk Card */}
      <div className="rounded-xl border border-white/10 bg-[#0B0E14]/90 p-5 backdrop-blur-xl shadow-2xl transition-all hover:border-green-500/30">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xs font-bold tracking-widest text-gray-400 uppercase">ML Model & Fleet Risk</h2>
          <Leaf size={16} className="text-green-400" />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <div className="text-3xl font-mono text-green-400 font-bold">{animatedAccuracy}%</div>
            <div className="text-[10px] text-gray-500 font-mono mt-1">XGBOOST + GNN ACCURACY</div>
          </div>

          <div className="text-right">
            <div className="text-xl font-mono text-orange-400 font-bold">{animatedAvgRisk}%</div>
            <div className="text-[10px] text-gray-500 font-mono mt-1">AVG THERMAL RISK</div>
          </div>
        </div>

        <div className="mt-4 flex justify-between text-[11px] font-mono text-gray-400 border-t border-white/5 pt-2">
          <span>Active Alerts: <strong className="text-orange-400">{modelStats.active_alerts ?? criticalRacks}</strong></span>
          <span>Hot Zones: <strong className="text-red-400">{criticalRacks}</strong></span>
        </div>
      </div>
    </>
  );
}
