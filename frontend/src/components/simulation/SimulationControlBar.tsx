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
    <div className="w-full rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md font-sans">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded bg-neutral-800 text-neutral-300 border border-neutral-700">
            <Sliders size={16} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold text-neutral-100 uppercase tracking-wide font-sans">Simulation Execution & Physics Control</h3>
              <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded-sm border ${
                config.isPaused 
                  ? 'bg-amber-950/80 text-amber-300 border-amber-800' 
                  : 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
              }`}>
                {config.isPaused ? 'PAUSED' : `RUNNING (${(1000 / config.simSpeedMs).toFixed(1)} Hz — ${config.simSpeedMs}ms step)`}
              </span>
            </div>
            <p className="text-[11px] text-neutral-400 font-mono mt-0.5">
              Adjust execution step frequency, freeze simulation state, or tune thermal dissipation curves.
            </p>
          </div>
        </div>

        {/* Quick Action Controls */}
        <div className="flex items-center gap-2">
          {/* Play/Pause Button */}
          <button
            onClick={() => togglePauseSimulation()}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-semibold transition-colors border ${
              config.isPaused
                ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800 hover:bg-emerald-900/80'
                : 'bg-amber-950/80 text-amber-300 border-amber-800 hover:bg-amber-900/80'
            }`}
          >
            {config.isPaused ? <Play size={13} className="fill-current" /> : <Pause size={13} className="fill-current" />}
            {config.isPaused ? 'Resume Simulation' : 'Pause Simulation'}
          </button>

          {/* Step Forward Button */}
          <button
            onClick={() => stepSimulation()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-semibold bg-zinc-800 text-zinc-200 border border-zinc-700 hover:bg-zinc-700 transition-colors"
            title="Step simulation forward by exactly 1 tick"
          >
            <SkipForward size={13} /> Step +1 Tick
          </button>

          {/* Toggle Expand Sliders */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="px-3 py-1.5 text-xs font-mono font-semibold rounded-md bg-neutral-800 text-neutral-300 border border-neutral-700 hover:bg-neutral-700 transition-colors"
          >
            {isExpanded ? 'Hide Physics Controls' : 'Physics Controls'}
          </button>
        </div>
      </div>

      {/* Expanded Control Sliders & Speed Selectors */}
      {isExpanded && (
        <div className="mt-4 pt-3.5 border-t border-neutral-800/80 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          
          {/* 1. Speed Control */}
          <div className="flex flex-col gap-2 bg-[#181b20] p-3 rounded-md border border-neutral-800">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-neutral-300 font-mono flex items-center gap-1.5">
                <FastForward size={13} /> Step Interval
              </span>
              <span className="text-xs font-mono font-bold text-neutral-200">
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
                    className={`py-1 text-[11px] font-mono font-semibold rounded-sm transition-colors ${
                      isActive
                        ? 'bg-zinc-700 text-neutral-100 border border-zinc-600 font-bold'
                        : 'bg-neutral-900 text-neutral-400 border border-neutral-800 hover:text-neutral-200'
                    }`}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. Base Thermal Heat Load */}
          <div className="flex flex-col gap-2 bg-[#181b20] p-3 rounded-md border border-neutral-800">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-neutral-300 font-mono flex items-center gap-1.5">
                <Thermometer size={13} /> Base Thermal Load
              </span>
              <span className="text-xs font-mono font-bold text-neutral-200">
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
              className="w-full accent-zinc-400 cursor-pointer h-1.5 bg-neutral-900 rounded-sm"
            />
            <div className="flex justify-between text-[10px] text-neutral-400 font-mono">
              <span>10% (Idle)</span>
              <span>50% (Baseline)</span>
              <span>100% (High)</span>
            </div>
          </div>

          {/* 3. Cooling Dissipation Rate */}
          <div className="flex flex-col gap-2 bg-[#181b20] p-3 rounded-md border border-neutral-800">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-neutral-300 font-mono flex items-center gap-1.5">
                <Wind size={13} /> Heat Dissipation
              </span>
              <span className="text-xs font-mono font-bold text-neutral-200">
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
              className="w-full accent-zinc-400 cursor-pointer h-1.5 bg-neutral-900 rounded-sm"
            />
            <div className="flex justify-between text-[10px] text-neutral-400 font-mono">
              <span>0.2x (Slow)</span>
              <span>1.0x (Standard)</span>
              <span>2.5x (Fast)</span>
            </div>
          </div>

          {/* 4. Thermal Multiplier & Workload Jitter */}
          <div className="flex flex-col gap-2 bg-[#181b20] p-3 rounded-md border border-neutral-800">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-neutral-300 font-mono flex items-center gap-1.5">
                <Activity size={13} /> Workload Variance
              </span>
              <span className="text-xs font-mono font-bold text-neutral-200">
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
              className="w-full accent-zinc-400 cursor-pointer h-1.5 bg-neutral-900 rounded-sm"
            />
            <div className="flex justify-between text-[10px] text-neutral-400 font-mono">
              <span>0% (Static)</span>
              <span>12% (Organic)</span>
              <span>40% (High)</span>
            </div>
          </div>

          {/* Action Trigger Row */}
          <div className="md:col-span-2 lg:col-span-4 flex flex-wrap items-center justify-between pt-2 border-t border-neutral-800/80 gap-2.5">
            <div className="flex items-center gap-2">
              <button
                onClick={() => injectGlobalSpike()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-red-950/70 text-red-300 border border-red-800 text-xs font-mono font-semibold hover:bg-red-900/70 transition-colors"
              >
                <Flame size={13} /> Inject Global Thermal Spike (All Racks)
              </button>

              <button
                onClick={() => injectRandomSpike()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-amber-950/70 text-amber-300 border border-amber-800 text-xs font-mono font-semibold hover:bg-amber-900/70 transition-colors"
              >
                <Zap size={13} /> Inject Random Rack Spike
              </button>
            </div>

            <button
              onClick={() => resetSimulation()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-neutral-800 text-neutral-300 border border-neutral-700 text-xs font-mono font-semibold hover:bg-neutral-700 transition-colors"
            >
              <RotateCcw size={13} /> Reset Simulation Defaults
            </button>
          </div>

        </div>
      )}
    </div>
  );
}
