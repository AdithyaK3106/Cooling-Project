import React from 'react';
import { DataCenterScene } from '../three/DataCenterScene';
import { SceneControlBar } from '../features/SceneControlBar';
import { useTelemetry } from '../services/telemetryApi';
import { getSimulatedTelemetry } from '../services/simulation';

export function ThermalMapTab() {
  const { data: telemetry } = useTelemetry();

  // Live time ticker
  const [now, setNow] = React.useState(new Date());
  React.useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
  const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' });

  // Calculate live average temperature and fleet risk score
  const sim = getSimulatedTelemetry();
  const racksList = telemetry?.racks || sim?.racks || [];
  const totalRacks = racksList.length || 25;
  
  const avgTemp = racksList.length > 0
    ? racksList.reduce((acc: number, r: any) => acc + (r.telemetry?.cpu_temp ?? 28.4), 0) / racksList.length
    : 28.4;

  const maxRisk = racksList.length > 0
    ? Math.max(...racksList.map((r: any) => r.risk_score ?? 0))
    : 0.18;

  const riskLabel = maxRisk > 0.55 ? 'HIGH' : maxRisk >= 0.35 ? 'ELEVATED' : 'LOW';
  const riskColor = maxRisk > 0.55 ? 'text-red-400' : maxRisk >= 0.35 ? 'text-amber-400' : 'text-emerald-400';

  return (
    <div className="relative h-full w-full bg-[#080a0f] overflow-hidden select-none">
      
      {/* 3D SCENE LAYER */}
      <div className="absolute inset-0 z-0">
        <DataCenterScene />
      </div>

      {/* OVERLAY HUD INTERFACE MATCHING REFERENCE EXACTLY */}
      <div className="pointer-events-none absolute inset-0 z-10 flex flex-col justify-between p-6">
        
        {/* TOP BAR: Brand Header (Left) | Controls (Center) | Live Clock & Systems Online (Right) */}
        <div className="flex items-start justify-between">
          
          {/* Top-Left: THERVO Branding matching reference */}
          <div className="flex flex-col gap-3 pointer-events-auto">
            <div>
              <div className="flex items-center gap-1 text-white font-black tracking-widest text-2xl font-sans">
                <span>THERVO</span>
                <span className="text-[11px] font-normal text-gray-400 -translate-y-2">™</span>
              </div>
              <div className="text-[9px] tracking-[0.22em] text-gray-400 font-semibold uppercase">
                AI-DRIVEN DATA CENTER COOLING
              </div>
            </div>
            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
                <span>Live Digital Twin</span>
              </div>
              <div className="text-[10px] font-bold tracking-wider text-gray-400 uppercase">
                RACK THERMAL MAP
              </div>
            </div>
          </div>

          {/* Top-Center: Perspective & Mode Switcher */}
          <div className="pointer-events-auto">
            <SceneControlBar />
          </div>

          {/* Top-Right: Clock, Systems Online, Mini Sparkline matching reference */}
          <div className="flex flex-col items-end gap-1.5 pointer-events-auto">
            <div className="text-right font-mono">
              <div className="text-sm font-bold text-white tracking-wider">{timeStr}</div>
              <div className="text-[10px] text-gray-400">{dateStr}</div>
            </div>
            <div className="flex items-center gap-2 text-xs font-semibold text-gray-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
              <span>Systems Online</span>
            </div>
            {/* Sleek SVG Sparkline */}
            <svg className="w-24 h-5 stroke-cyan-400 fill-none" viewBox="0 0 100 20">
              <path d="M 0 14 Q 25 3, 50 12 T 100 8" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </div>
        </div>



        {/* BOTTOM BAR: Metric Summary Card (Left) | Thermal State Legend (Right) */}
        <div className="flex items-end justify-between">
          
          {/* Bottom-Left: Total Racks | Avg Temperature | Risk Score */}
          <div className="pointer-events-auto rounded-xl border border-white/10 bg-[#0B0E14]/85 px-5 py-3 backdrop-blur-xl shadow-2xl flex items-center gap-6 text-gray-200">
            <div>
              <div className="text-[9px] font-bold tracking-wider text-gray-400 uppercase">Total Racks</div>
              <div className="text-xl font-mono font-bold text-white">{totalRacks}</div>
            </div>
            <div className="h-7 w-[1px] bg-white/10" />
            <div>
              <div className="text-[9px] font-bold tracking-wider text-gray-400 uppercase">Avg Temperature</div>
              <div className="text-xl font-mono font-bold text-white">{avgTemp.toFixed(1)}°C</div>
            </div>
            <div className="h-7 w-[1px] bg-white/10" />
            <div>
              <div className="text-[9px] font-bold tracking-wider text-gray-400 uppercase">Risk Score</div>
              <div className={`text-xl font-mono font-bold ${riskColor}`}>{riskLabel}</div>
            </div>
          </div>

          {/* Bottom-Right: Thermal State Legend matching reference */}
          <div className="pointer-events-auto rounded-xl border border-white/10 bg-[#0B0E14]/85 px-5 py-3.5 backdrop-blur-xl shadow-2xl flex flex-col gap-1.5 text-xs text-gray-300">
            <div className="text-[10px] font-bold tracking-wider text-gray-400 uppercase mb-1">Thermal State</div>
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#06b6d4]" />
              <span>Normal ( &lt; 30°C )</span>
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <span className="h-2 w-2 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]" />
              <span>Elevated ( 30 - 35°C )</span>
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <span className="h-2 w-2 rounded-full bg-red-500 shadow-[0_0_6px_#ef4444]" />
              <span>High Risk ( &gt; 35°C )</span>
            </div>
          </div>

        </div>

      </div>
      
    </div>
  );
}
