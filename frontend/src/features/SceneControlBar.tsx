import { Thermometer, Wind, AlertTriangle, Box, Eye } from 'lucide-react';

// Note: In a real app we'd add these modes to the zustand store.
// For now, we'll keep it visually accurate.
export function SceneControlBar() {
  return (
    <div className="flex gap-4 pointer-events-auto">
      
      {/* View Perspective Controls */}
      <div className="flex items-center rounded-full border border-white/10 bg-[#0B0E14]/80 p-1 backdrop-blur-xl shadow-lg">
        <button className="flex items-center gap-2 rounded-full bg-cyan-500/20 px-4 py-1.5 text-xs font-bold text-cyan-400">
          <Box size={14} /> 3D
        </button>
        <button className="flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold text-gray-400 hover:text-white transition-colors">
          <Eye size={14} /> TOP
        </button>
      </div>

      {/* Layer Controls */}
      <div className="flex items-center rounded-full border border-white/10 bg-[#0B0E14]/80 p-1 backdrop-blur-xl shadow-lg">
        <button className="flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold text-orange-400 bg-orange-500/10 transition-colors">
          <Thermometer size={14} /> THERMAL
        </button>
        <button className="flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold text-gray-400 hover:text-white transition-colors">
          <Wind size={14} /> AIRFLOW
        </button>
        <button className="flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold text-gray-400 hover:text-white transition-colors">
          <AlertTriangle size={14} /> RISK
        </button>
      </div>

    </div>
  );
}
