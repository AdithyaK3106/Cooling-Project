import { useState, useEffect } from 'react';
import { 
  Play, Pause, Flame, Zap, RotateCcw, 
  Thermometer, Activity, Sliders, Wind, FastForward, SkipForward
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
    <div className="w-full rounded-xl border border-cyan-500/30 bg-[#0B0E14]/95 p-4 backdrop-blur shadow-2xl transition-all">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${config.isPaused ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'}`}>
            <Sliders size={18} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-white tracking-wide">Simulation Physics & Speed Master Controller</h3>
              <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded-full border ${
                config.isPaused 
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/40' 
                  : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40 animate-pulse'
              }`}>
                {config.isPaused ? 'PAUSED' : `RUNNING (${(1000 / config.simSpeedMs).toFixed(1)} Hz — ${config.simSpeedMs}ms step)`}
              </span>
            </div>
            <p className="text-xs text-gray-400">
              Set exact simulation step speed (up to 5 seconds per tick), freeze physics, step tick-by-tick, or adjust thermal heat load.
            </p>
          </div>
        </div>

        {/* Quick Action Controls */}
        <div className="flex items-center gap-2">
          {/* Play/Pause Button */}
          <button
            onClick={() => togglePauseSimulation()}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all shadow-md active:scale-95 border ${
              config.isPaused
                ? 'bg-emerald-600/30 text-emerald-300 border-emerald-500/50 hover:bg-emerald-600/50'
                : 'bg-amber-600/30 text-amber-300 border-amber-500/50 hover:bg-amber-600/50'
            }`}
          >
            {config.isPaused ? <Play size={14} className="fill-current" /> : <Pause size={14} className="fill-current" />}
            {config.isPaused ? 'Resume Simulation' : 'Pause Simulation'}
          </button>

          {/* Step Forward Button */}
          <button
            onClick={() => stepSimulation()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-cyan-600/30 text-cyan-300 border border-cyan-500/50 hover:bg-cyan-600/50 transition-all shadow-md active:scale-95"
            title="Step simulation forward by exactly 1 tick"
          >
            <SkipForward size={14} /> Step +1 Tick
          </button>

          {/* Toggle Expand Sliders */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-white/5 text-gray-300 border border-white/10 hover:bg-white/10 transition-colors"
          >
            {isExpanded ? 'Collapse Sliders' : 'Tune Physics Controls'}
          </button>
        </div>
      </div>

      {/* Expanded Control Sliders & Speed Selectors */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          
          {/* 1. Speed Control */}
          <div className="flex flex-col gap-2 bg-white/5 p-3 rounded-lg border border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-400 flex items-center gap-1.5">
                <FastForward size={14} /> Simulation Speed
              </span>
              <span className="text-xs font-mono font-bold text-gray-300">
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
                    className={`py-1 text-xs font-mono font-bold rounded transition-all ${
                      isActive
                        ? 'bg-cyan-500 text-black shadow-md shadow-cyan-500/30'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                    }`}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. Base Thermal Heat Load */}
          <div className="flex flex-col gap-2 bg-white/5 p-3 rounded-lg border border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-amber-400 flex items-center gap-1.5">
                <Thermometer size={14} /> Base Thermal Load
              </span>
              <span className="text-xs font-mono font-bold text-amber-300">
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
              className="w-full accent-amber-500 cursor-pointer h-1.5 bg-white/10 rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-gray-400 font-mono">
              <span>10% (Idle)</span>
              <span>50% (Normal)</span>
              <span>100% (Extreme)</span>
            </div>
          </div>

          {/* 3. Cooling Dissipation Rate */}
          <div className="flex flex-col gap-2 bg-white/5 p-3 rounded-lg border border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-blue-400 flex items-center gap-1.5">
                <Wind size={14} /> Cooling Dissipation
              </span>
              <span className="text-xs font-mono font-bold text-blue-300">
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
              className="w-full accent-blue-500 cursor-pointer h-1.5 bg-white/10 rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-gray-400 font-mono">
              <span>0.2x (Slow Cool)</span>
              <span>1.0x (Standard)</span>
              <span>2.5x (Rapid)</span>
            </div>
          </div>

          {/* 4. Thermal Multiplier & Workload Jitter */}
          <div className="flex flex-col gap-2 bg-white/5 p-3 rounded-lg border border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-purple-400 flex items-center gap-1.5">
                <Activity size={14} /> Workload Volatility
              </span>
              <span className="text-xs font-mono font-bold text-purple-300">
                {config.noise}% Jitter
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="40"
              step="2"
              value={config.noise}
              onChange={(e) => setSimulationParams(config.load, Number(e.target.value), config.dissipationRate, config.thermalMultiplier)}
              className="w-full accent-purple-500 cursor-pointer h-1.5 bg-white/10 rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-gray-400 font-mono">
              <span>0% (Static)</span>
              <span>12% (Organic)</span>
              <span>40% (Volatile)</span>
            </div>
          </div>

          {/* Action Trigger Row */}
          <div className="md:col-span-2 lg:col-span-4 flex flex-wrap items-center justify-between pt-2 border-t border-white/5 gap-3">
            <div className="flex items-center gap-2">
              <button
                onClick={() => injectGlobalSpike()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-600/30 text-red-300 border border-red-500/40 text-xs font-bold hover:bg-red-600/50 transition-all shadow-md active:scale-95"
              >
                <Flame size={14} /> Inject Global Thermal Spike (All Racks)
              </button>

              <button
                onClick={() => injectRandomSpike()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-orange-600/20 text-orange-300 border border-orange-500/30 text-xs font-bold hover:bg-orange-600/40 transition-all shadow-md active:scale-95"
              >
                <Zap size={14} /> Inject Random Rack Spike
              </button>
            </div>

            <button
              onClick={() => resetSimulation()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 text-gray-300 border border-gray-700 text-xs font-bold hover:bg-gray-700 transition-all active:scale-95"
            >
              <RotateCcw size={14} /> Reset Simulation Defaults
            </button>
          </div>

        </div>
      )}
    </div>
  );
}
