import { useState } from 'react';
import { Play, Settings2 } from 'lucide-react';
import { useUpdateSimulationControls } from '../services/controlApi';

export function CompactBottomBar() {
  const mutation = useUpdateSimulationControls();
  
  // Local state for sliders (to avoid constant API spam)
  const [load, setLoad] = useState(50);
  const [tempOffset, setTempOffset] = useState(0);

  const handleApply = () => {
    mutation.mutate({
      simulated_load: load,
      ambient_temp_offset: tempOffset,
      trigger_spike: load > 80,
      mode: 'DATA_CENTER_SIMULATION'
    });
  };

  return (
    <div className="flex h-full w-full gap-4">
      
      {/* LEFT: Simulation Controls */}
      <div className="flex w-[400px] items-center justify-between rounded-xl border border-white/10 bg-[#0B0E14]/90 p-4 backdrop-blur-xl shadow-2xl">
        
        <div className="flex gap-4 items-center">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Sim Load</span>
            <input 
              type="range" 
              min="10" max="100" 
              value={load} 
              onChange={(e) => setLoad(parseInt(e.target.value))}
              className="w-24 accent-cyan-500" 
            />
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Thermal Off.</span>
            <input 
              type="range" 
              min="-10" max="20" 
              value={tempOffset} 
              onChange={(e) => setTempOffset(parseInt(e.target.value))}
              className="w-24 accent-orange-500" 
            />
          </div>
        </div>

        <button 
          onClick={handleApply}
          className="flex h-10 items-center gap-2 rounded-lg bg-cyan-600/20 px-4 text-xs font-bold text-cyan-400 border border-cyan-500/30 hover:bg-cyan-600/40 transition-colors"
        >
          <Play size={14} /> APPLY
        </button>

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
