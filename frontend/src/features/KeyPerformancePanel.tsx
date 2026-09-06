import { useTelemetry } from '../services/telemetryApi';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { Leaf, Zap, Activity } from 'lucide-react';

export function KeyPerformancePanel() {
  const { data: telemetry } = useTelemetry();

  // Mock aggregated metrics based on telemetry
  const racks = telemetry?.racks || [];
  const avgTemp = racks.reduce((acc: number, r: any) => acc + r.telemetry.cpu_temp, 0) / (racks.length || 1) || 45;
  const totalPower = racks.reduce((acc: number, r: any) => acc + r.telemetry.power_draw, 0) || 1200;
  
  // Calculate a fake PUE (Power Usage Effectiveness) - 1.0 is perfect, 1.5 is average
  // Assuming cooling scales with temperature
  const pue = 1.0 + (avgTemp / 100) * 0.5;
  const animatedPUE = useAnimatedNumber(pue, 800, (v: number) => v.toFixed(2));

  // Sustainability score (0-100)
  const greenScore = Math.max(0, Math.min(100, 100 - (pue - 1.0) * 100));
  const animatedGreen = useAnimatedNumber(greenScore, 800, (v: number) => Math.round(v).toString());

  return (
    <div className="flex flex-col gap-4">
      
      {/* PUE Card */}
      <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur-xl shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-sm font-bold tracking-widest text-gray-400 uppercase">Facility PUE</h2>
          <Zap size={16} className="text-cyan-400" />
        </div>
        <div className="flex items-end gap-3">
          <div className="text-4xl font-mono text-cyan-400 font-light">{animatedPUE}</div>
          <div className="mb-1 text-sm text-gray-500 font-mono">1.0 TARGET</div>
        </div>
        
        {/* Simple inline visualization (gradient bar) */}
        <div className="mt-4 h-1.5 w-full rounded-full bg-gray-800 overflow-hidden flex">
          <div className="h-full bg-gradient-to-r from-cyan-400 to-blue-500" style={{ width: `${(2.0 - pue) * 100}%` }} />
        </div>
      </div>

      {/* Cooling Energy Card */}
      <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur-xl shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-sm font-bold tracking-widest text-gray-400 uppercase">Cooling Draw</h2>
          <Activity size={16} className="text-blue-400" />
        </div>
        <div className="flex items-end gap-2">
          <div className="text-3xl font-mono text-blue-400 font-light">{useAnimatedNumber(totalPower * 0.3, 800, (v: number) => Math.round(v).toString())}</div>
          <div className="mb-1 text-sm text-gray-500 font-mono">kW</div>
        </div>
        
        {/* Fake sparkline made of CSS bars for aesthetics */}
        <div className="mt-5 flex items-end justify-between h-8 gap-1 opacity-70">
          {[40, 45, 30, 50, 60, 45, 55, 35, 40, 50, 65, 45].map((val, i) => (
            <div key={i} className="w-full bg-blue-500/50 rounded-t-sm" style={{ height: `${val}%` }} />
          ))}
        </div>
      </div>

      {/* Green Compliance */}
      <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur-xl shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-sm font-bold tracking-widest text-gray-400 uppercase">Green Score</h2>
          <Leaf size={16} className="text-green-400" />
        </div>
        
        <div className="flex items-center gap-5">
          <div className="relative w-16 h-16 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path className="text-gray-800" strokeWidth="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              <path className="text-green-400 transition-all duration-1000 ease-out" strokeWidth="3" strokeDasharray={`${greenScore}, 100`} stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            </svg>
            <div className="absolute text-lg font-mono text-white">{animatedGreen}</div>
          </div>
          <div className="text-xs text-gray-400 leading-relaxed">
            Carbon footprint is well within compliance margins.
          </div>
        </div>
      </div>

    </div>
  );
}
