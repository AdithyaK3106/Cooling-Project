import { Thermometer, Wind, AlertTriangle, Box, Eye } from 'lucide-react';
import { useUiStore } from '../stores/uiStore';

export function SceneControlBar() {
  const { perspective, setPerspective, activeLayer, setActiveLayer } = useUiStore();

  return (
    <div className="flex gap-4 pointer-events-auto">
      
      {/* View Perspective Controls */}
      <div className="flex items-center rounded-full border border-white/10 bg-[#0B0E14]/80 p-1 backdrop-blur-xl shadow-lg">
        <button 
          onClick={() => setPerspective('3D')}
          className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold transition-colors ${perspective === '3D' ? 'bg-cyan-500/20 text-cyan-400' : 'text-gray-400 hover:text-white'}`}
        >
          <Box size={14} /> 3D
        </button>
        <button 
          onClick={() => setPerspective('TOP')}
          className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold transition-colors ${perspective === 'TOP' ? 'bg-cyan-500/20 text-cyan-400' : 'text-gray-400 hover:text-white'}`}
        >
          <Eye size={14} /> TOP
        </button>
        <button 
          onClick={() => setPerspective('SIDE')}
          className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold transition-colors ${perspective === 'SIDE' ? 'bg-cyan-500/20 text-cyan-400' : 'text-gray-400 hover:text-white'}`}
        >
          <Eye size={14} /> SIDE
        </button>
      </div>

      {/* Layer Controls */}
      <div className="flex items-center rounded-full border border-white/10 bg-[#0B0E14]/80 p-1 backdrop-blur-xl shadow-lg">
        <button 
          onClick={() => setActiveLayer('THERMAL')}
          className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold transition-colors ${activeLayer === 'THERMAL' ? 'bg-orange-500/20 text-orange-400' : 'text-gray-400 hover:text-white'}`}
        >
          <Thermometer size={14} /> THERMAL
        </button>
        <button 
          onClick={() => setActiveLayer('AIRFLOW')}
          className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold transition-colors ${activeLayer === 'AIRFLOW' ? 'bg-blue-500/20 text-blue-400' : 'text-gray-400 hover:text-white'}`}
        >
          <Wind size={14} /> AIRFLOW
        </button>
        <button 
          onClick={() => setActiveLayer('RISK')}
          className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold transition-colors ${activeLayer === 'RISK' ? 'bg-red-500/20 text-red-400' : 'text-gray-400 hover:text-white'}`}
        >
          <AlertTriangle size={14} /> RISK
        </button>
      </div>

    </div>
  );
}
