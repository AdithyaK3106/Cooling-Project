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
  const animatedCoolingPower = useAnimatedNumber(totalPower * 0.35 / 1000, 600, (v: number) => v.toFixed(2));
  const animatedAccuracy = useAnimatedNumber(modelStats.accuracy || 96.4, 600, (v: number) => v.toFixed(1));
  const animatedAvgRisk = useAnimatedNumber(avgRisk, 600, (v: number) => v.toFixed(1));

  const cardBaseClass = "rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-4.5 backdrop-blur-md shadow-2xl flex flex-col justify-between font-sans hover:border-white/[0.15] transition-all";

  return (
    <>
      {/* 1. Facility PUE Card */}
      <div className={cardBaseClass}>
        <div>
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                <Zap size={14} />
              </div>
              <h2 className="text-xs font-mono font-bold tracking-wider text-gray-300 uppercase">Facility PUE</h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/[0.05] text-gray-400 border border-white/[0.08]">
              REAL-TIME
            </span>
          </div>

          <div className="flex items-baseline justify-between mt-1">
            <div>
              <div className="text-3xl font-mono font-bold text-white tracking-tight flex items-baseline gap-1">
                {animatedPUE}
                <span className="text-xs text-gray-400 font-normal">PUE</span>
              </div>
              <div className="text-[10px] font-mono text-gray-400 mt-0.5">TARGET: 1.00 BASELINE</div>
            </div>

            <div className="text-right font-mono">
              <div className="text-[10px] text-gray-400 uppercase">EFFICIENCY</div>
              <div className="text-base font-bold text-cyan-400">{(100 / pue).toFixed(1)}%</div>
            </div>
          </div>
        </div>
        
        <div className="mt-4">
          <div className="h-1.5 w-full rounded-full bg-[#181d28] overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 shadow-[0_0_8px_rgba(6,182,212,0.6)] transition-all duration-500 rounded-full" 
              style={{ width: `${Math.min(100, Math.max(10, (2.0 - pue) * 100))}%` }} 
            />
          </div>
          <div className="flex justify-between text-[9px] font-mono text-gray-400 mt-1">
            <span>Optimal (1.0)</span>
            <span>Current ({animatedPUE})</span>
            <span>Threshold (2.0)</span>
          </div>
        </div>
      </div>

      {/* 2. Cooling Power Draw Card */}
      <div className={cardBaseClass}>
        <div>
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-blue-500/10 text-blue-400 border border-blue-500/30">
                <Activity size={14} />
              </div>
              <h2 className="text-xs font-mono font-bold tracking-wider text-gray-300 uppercase">Cooling Power Draw</h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-semibold">
              ACTIVE
            </span>
          </div>

          <div className="flex items-baseline justify-between mt-1">
            <div>
              <div className="text-3xl font-mono font-bold text-white tracking-tight flex items-baseline gap-1">
                {animatedCoolingPower}
                <span className="text-xs text-gray-400 font-normal">MW</span>
              </div>
              <div className="text-[10px] font-mono text-gray-400 mt-0.5">ESTIMATED COOLING LOAD</div>
            </div>

            <div className="text-right font-mono">
              <div className="text-[10px] text-gray-400 uppercase">COOLING RACKS</div>
              <div className="text-base font-bold text-cyan-300">{coolingRacks} <span className="text-xs text-gray-400">/ {racks.length || 25}</span></div>
            </div>
          </div>
        </div>
        
        {/* Sparkline distribution of rack risk */}
        <div className="mt-4">
          <div className="flex items-end justify-between h-5 gap-1">
            {racks.slice(0, 25).map((r: any, i: number) => {
              const rRisk = r.risk_score || 0;
              const isCrit = rRisk >= 0.75;
              const isHigh = rRisk >= 0.55;
              const isMed = rRisk >= 0.35;
              
              let barBg = 'bg-cyan-500/70 hover:bg-cyan-400';
              if (isCrit) barBg = 'bg-red-500 shadow-[0_0_6px_#ef4444]';
              else if (isHigh) barBg = 'bg-orange-500';
              else if (isMed) barBg = 'bg-amber-400';

              return (
                <div 
                  key={i} 
                  title={`Rack ${r.id}: ${(rRisk * 100).toFixed(0)}% Risk`}
                  className={`w-full rounded-t-sm transition-all duration-300 cursor-pointer ${barBg}`} 
                  style={{ height: `${Math.max(15, rRisk * 100)}%` }} 
                />
              );
            })}
          </div>
          <div className="flex justify-between text-[9px] font-mono text-gray-400 mt-1">
            <span>Sector A</span>
            <span>Cluster Heat Spread</span>
            <span>Sector E</span>
          </div>
        </div>
      </div>

      {/* 3. ML Model & Fleet Risk Card */}
      <div className={cardBaseClass}>
        <div>
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <Leaf size={14} />
              </div>
              <h2 className="text-xs font-mono font-bold tracking-wider text-gray-300 uppercase">ML Model & Fleet Risk</h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 font-semibold">
              GNN + XGB
            </span>
          </div>
          
          <div className="flex items-baseline justify-between mt-1">
            <div>
              <div className="text-3xl font-mono font-bold text-emerald-400 tracking-tight flex items-baseline gap-1">
                {animatedAccuracy}%
              </div>
              <div className="text-[10px] font-mono text-gray-400 mt-0.5">PREDICTIVE ACCURACY</div>
            </div>

            <div className="text-right font-mono">
              <div className="text-[10px] text-gray-400 uppercase">AVG THERMAL RISK</div>
              <div className={`text-2xl font-bold ${avgRisk > 55 ? 'text-red-400' : avgRisk > 35 ? 'text-amber-400' : 'text-cyan-300'}`}>
                {animatedAvgRisk}%
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 pt-2 border-t border-white/[0.08] flex justify-between items-center text-[11px] font-mono">
          <div className="flex items-center gap-1.5">
            <span className="text-gray-400">Active Alerts:</span>
            <span className="px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 font-bold">
              {modelStats.active_alerts ?? criticalRacks}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-gray-400">Hot Zones:</span>
            <span className="px-1.5 py-0.2 rounded bg-red-500/10 text-red-300 border border-red-500/30 font-bold">
              {criticalRacks}
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
