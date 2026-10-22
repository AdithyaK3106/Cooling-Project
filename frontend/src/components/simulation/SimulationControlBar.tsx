import { useState, useEffect } from 'react';
import { 
  Play, Pause, Flame, Zap, RotateCcw, 
  Thermometer, Activity, Sliders, Wind, FastForward, SkipForward, ChevronDown, ChevronUp
} from 'lucide-react';
import { 
  getSimulationConfig, 
  setSimulationSpeed, 
  togglePauseSimulation, 
  stepSimulation,
  setSimulationParams, 
  injectGlobalSpike, 
  injectRandomSpike, 
  resetSimulation,
  subscribeSimulationConfig 
} from '../../services/simulation';
export function SimulationControlBar() {
  const [config, setConfig] = useState(getSimulationConfig());
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const unsubscribe = subscribeSimulationConfig(() => {
      setConfig(getSimulationConfig());
    });
    return unsubscribe;
  }, []);

  const speeds = [
    { label: '0.1x', ms: 5000, desc: '5.0s per tick (Super Slow)' },
    { label: '0.25x', ms: 2000, desc: '2.0s per tick (Very Slow)' },
    { label: '0.5x', ms: 1000, desc: '1.0s per tick (Slow)' },
    { label: '1.0x', ms: 500, desc: '0.5s per tick (Standard)' },
    { label: '2.0x', ms: 250, desc: '0.25s per tick (Fast)' },
    { label: '4.0x', ms: 100, desc: '0.1s per tick (Turbo)' },
  ];

  return (
    <div className="w-full rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-4 backdrop-blur-md shadow-2xl font-sans">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_12px_rgba(6,182,212,0.2)]">
            <Sliders size={16} />
          </div>
          <div>
            <div className="flex items-center gap-2.5 flex-wrap">
              <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider">Simulation Execution & Physics Control</h3>
              <span className={`px-2.5 py-0.5 text-[10px] font-mono font-bold rounded-full border ${
                config.isPaused 
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/40 shadow-[0_0_8px_rgba(245,158,11,0.2)]'
                  : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40 shadow-[0_0_8px_rgba(16,185,129,0.2)]'
              }`}>
                {config.isPaused ? 'PAUSED' : `RUNNING (${(1000 / config.simSpeedMs).toFixed(1)} Hz • ${config.simSpeedMs}ms)`}
              </span>
            </div>
            <p className="text-[11px] text-gray-400 font-mono mt-0.5">
              Live physics engine: adjust execution step frequency, freeze simulation state, or tune thermal dissipation curves.
            </p>
          </div>
        </div>

        {/* Quick Action Controls */}
        <div className="flex items-center gap-2">
          {/* Play/Pause Button */}
          <button
            onClick={() => togglePauseSimulation()}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all border shadow-sm ${
              config.isPaused
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/30 shadow-[0_0_10px_rgba(16,185,129,0.2)]'
                : 'bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30 shadow-[0_0_10px_rgba(245,158,11,0.2)]'
            }`}
          >
            {config.isPaused ? <Play size={13} className="fill-current" /> : <Pause size={13} className="fill-current" />}
            {config.isPaused ? 'Resume Sim' : 'Pause Sim'}
          </button>

          {/* Step Forward Button */}
          <button
            onClick={() => stepSimulation()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-white/[0.05] text-gray-200 border border-white/10 hover:bg-white/[0.1] hover:border-white/20 transition-all"
            title="Step simulation forward by exactly 1 tick"
          >
            <SkipForward size={13} /> Step +1 Tick
          </button>

          {/* Toggle Expand Sliders */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 px-3 py-1.5 text-xs font-mono font-semibold rounded-lg bg-white/[0.05] text-gray-300 border border-white/10 hover:bg-white/[0.1] hover:text-white transition-all"
          >
            {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            {isExpanded ? 'Hide Physics' : 'Physics Controls'}
          </button>
        </div>
      </div>

      {/* Expanded Control Sliders & Speed Selectors */}
      {isExpanded && (
        <div className="mt-4 pt-3.5 border-t border-white/[0.08] grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
          
          {/* 1. Speed Control */}
          <div className="flex flex-col gap-2 bg-[#10141D] p-3 rounded-lg border border-white/[0.06]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-300 font-mono flex items-center gap-1.5">
                <FastForward size={13} className="text-cyan-400" /> Step Interval
              </span>
              <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/20">
                {config.simSpeedMs}ms / tick
              </span>
            </div>
            <div className="grid grid-cols-6 gap-1 mt-1">
              {speeds.map((s) => {
                const isActive = config.simSpeedMs === s.ms;
                return (
                  <button
                    key={s.label}
                    onClick={() => setSimulationSpeed(s.ms)}
                    title={s.desc}
                    className={`py-1 text-[11px] font-mono font-semibold rounded transition-all ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 font-bold shadow-[0_0_8px_rgba(6,182,212,0.3)]'
                        : 'bg-[#151924] text-gray-400 border border-white/[0.05] hover:text-gray-200 hover:bg-white/[0.05]'
                    }`}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. Base Thermal Heat Load */}
          <div className="flex flex-col gap-2 bg-[#10141D] p-3 rounded-lg border border-white/[0.06]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-300 font-mono flex items-center gap-1.5">
                <Thermometer size={13} className="text-amber-400" /> Base Thermal Load
              </span>
              <span className="text-xs font-mono font-bold text-amber-300 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">
                {config.load}%
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={config.load}
              onChange={(e) => setSimulationParams(Number(e.target.value), config.noise, config.dissipationRate, config.thermalMultiplier)}
              className="w-full h-1.5 bg-[#181d28] rounded-sm"
            />
            <div className="flex justify-between text-[10px] text-gray-400 font-mono">
              <span>10% (Idle)</span>
              <span>50% (Baseline)</span>
              <span>100% (High)</span>
            </div>
          </div>

          {/* 3. Cooling Dissipation Rate */}
          <div className="flex flex-col gap-2 bg-[#10141D] p-3 rounded-lg border border-white/[0.06]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-300 font-mono flex items-center gap-1.5">
                <Wind size={13} className="text-blue-400" /> Heat Dissipation
              </span>
              <span className="text-xs font-mono font-bold text-blue-300 bg-blue-500/10 px-1.5 py-0.5 rounded border border-blue-500/20">
                {config.dissipationRate.toFixed(1)}x Speed
              </span>
            </div>
            <input
              type="range"
              min="0.2"
              max="2.5"
              step="0.1"
              value={config.dissipationRate}
              onChange={(e) => setSimulationParams(config.load, config.noise, Number(e.target.value), config.thermalMultiplier)}
              className="w-full h-1.5 bg-[#181d28] rounded-sm"
            />
            <div className="flex justify-between text-[10px] text-gray-400 font-mono">
              <span>0.2x (Slow)</span>
              <span>1.0x (Standard)</span>
              <span>2.5x (Fast)</span>
            </div>
          </div>

          {/* 4. Thermal Multiplier & Workload Jitter */}
          <div className="flex flex-col gap-2 bg-[#10141D] p-3 rounded-lg border border-white/[0.06]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-300 font-mono flex items-center gap-1.5">
                <Activity size={13} className="text-purple-400" /> Workload Jitter
              </span>
              <span className="text-xs font-mono font-bold text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded border border-purple-500/20">
                {config.noise}% Variance
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="40"
              step="2"
              value={config.noise}
              onChange={(e) => setSimulationParams(config.load, Number(e.target.value), config.dissipationRate, config.thermalMultiplier)}
              className="w-full h-1.5 bg-[#181d28] rounded-sm"
            />
            <div className="flex justify-between text-[10px] text-gray-400 font-mono">
              <span>0% (Static)</span>
              <span>12% (Organic)</span>
              <span>40% (High)</span>
            </div>
          </div>

          {/* Action Trigger Row */}
          <div className="md:col-span-2 lg:col-span-4 flex flex-wrap items-center justify-between pt-2.5 border-t border-white/[0.06] gap-2.5">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => injectGlobalSpike()}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-red-500/15 text-red-300 border border-red-500/30 text-xs font-mono font-semibold hover:bg-red-500/25 hover:border-red-500/50 transition-all shadow-[0_0_10px_rgba(239,68,68,0.15)]"
              >
                <Flame size={13} /> Inject Global Thermal Spike (All Racks)
              </button>

              <button
                onClick={() => injectRandomSpike()}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-amber-500/15 text-amber-300 border border-amber-500/30 text-xs font-mono font-semibold hover:bg-amber-500/25 hover:border-amber-500/50 transition-all shadow-[0_0_10px_rgba(245,158,11,0.15)]"
              >
                <Zap size={13} /> Inject Random Rack Spike
              </button>
            </div>

            <button
              onClick={() => resetSimulation()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.05] text-gray-300 border border-white/10 text-xs font-mono font-semibold hover:bg-white/[0.1] hover:text-white transition-all"
            >
              <RotateCcw size={13} /> Reset Simulation Defaults
            </button>
          </div>

        </div>
      )}
    </div>
  );
}
