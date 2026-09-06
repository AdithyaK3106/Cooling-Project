import { useState } from 'react';
import { Play, Settings2 } from 'lucide-react';
import { useUpdateSimulationControls, useInjectSpike } from '../services/controlApi';

export function CompactBottomBar() {
  const simMutation = useUpdateSimulationControls();
  const spikeMutation = useInjectSpike();
  
  // Local state for sliders (to avoid constant API spam)
  const [load, setLoad] = useState(35);
  const [noise, setNoise] = useState(12);

  const handleApply = () => {
    simMutation.mutate({ load, noise });
  };

  return (
    <div className="flex h-full w-full gap-4">
      
      {/* LEFT: Simulation Controls */}
      <div className="flex w-full items-center justify-between rounded-xl border border-white/10 bg-[#0B0E14]/90 p-4 backdrop-blur-xl shadow-2xl">
        
        <div className="flex gap-4 items-center flex-1">
          <div className="flex flex-col gap-1 flex-1">
            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Sim Load: {load}%</span>
            <input 
              type="range" 
              min="10" max="100" 
              value={load} 
              onChange={(e) => setLoad(parseInt(e.target.value))}
              className="w-full accent-cyan-500" 
            />
          </div>

          <div className="flex flex-col gap-1 flex-1">
            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Thermal Noise: {noise}%</span>
            <input 
              type="range" 
              min="0" max="30" 
              value={noise} 
              onChange={(e) => setNoise(parseInt(e.target.value))}
              className="w-full accent-orange-500" 
            />
          </div>
        </div>

        <div className="flex gap-2 ml-6">
          <button 
            onClick={handleApply}
            className="flex h-10 items-center gap-2 rounded-lg bg-cyan-600/20 px-4 text-xs font-bold text-cyan-400 border border-cyan-500/30 hover:bg-cyan-600/40 transition-colors"
          >
            <Play size={14} /> APPLY
          </button>
          
          <button 
            onClick={() => spikeMutation.mutate()}
            className="flex h-10 items-center gap-2 rounded-lg bg-red-600/20 px-4 text-xs font-bold text-red-400 border border-red-500/30 hover:bg-red-600/40 transition-colors"
          >
            ⚡ INJECT SPIKE
          </button>
        </div>

      </div>

      {/* CENTER: Events Ticker */}
      <div className="flex flex-1 items-center justify-between rounded-xl border border-white/10 bg-[#0B0E14]/90 p-4 backdrop-blur-xl shadow-2xl overflow-hidden">
        <div className="flex items-center gap-6">
          <div className="text-xs font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2 border-r border-white/10 pr-6">
            <Settings2 size={16} /> SYSTEM LOG
          </div>
          
          <div className="flex items-center gap-8 text-xs font-mono">
            <div className="flex items-center gap-2 text-gray-300">
              <span className="w-2 h-2 rounded-full bg-cyan-500" />
              Auto cooling activated <span className="text-gray-500">— Rack A07</span>
            </div>
            <div className="flex items-center gap-2 text-gray-300">
              <span className="w-2 h-2 rounded-full bg-orange-500" />
              Temperature rising <span className="text-gray-500">— Rack B12</span>
            </div>
            <div className="flex items-center gap-2 text-gray-300">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              Optimal state reached <span className="text-gray-500">— Rack C03</span>
            </div>
          </div>
        </div>
        
        <button className="text-[10px] uppercase font-bold text-cyan-400 hover:text-cyan-300 tracking-widest">
          VIEW ALL
        </button>
      </div>

    </div>
  );
}
