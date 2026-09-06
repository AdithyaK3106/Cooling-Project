import { KeyPerformancePanel } from '../features/KeyPerformancePanel';
import { CompactBottomBar } from '../features/CompactBottomBar';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';

export function OverviewTab() {
  const { data: telemetry } = useTelemetry();
  const racks = telemetry?.racks || [];
  const { selectedRackId, setSelectedRackId } = useUiStore();

  return (
    <div className="flex h-full w-full flex-col gap-6 p-6 overflow-y-auto">
      
      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KeyPerformancePanel />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-[400px]">
        {/* Left/Center: Flat Floor Plan Grid */}
        <div className="lg:col-span-2 rounded-xl border border-white/10 bg-[#0B0E14]/80 p-6 backdrop-blur shadow-2xl flex flex-col">
          <div className="mb-6">
            <h2 className="text-lg font-bold text-white tracking-wide">Data Center Floor Plan (Flat)</h2>
            <p className="text-xs text-gray-400">Select a rack to view details</p>
          </div>
          
          <div className="flex-1 grid grid-cols-5 gap-4 content-start">
            {racks.map((rack) => {
              const isDanger = rack.risk_score > 0.7;
              const isWarning = rack.risk_score > 0.4 && !isDanger;
              const bgClass = isDanger ? 'bg-red-500/20 border-red-500/50' : 
                              isWarning ? 'bg-orange-500/20 border-orange-500/50' : 
                              'bg-green-500/10 border-green-500/30';
              
              const isSelected = selectedRackId === rack.id;
              const selectClass = isSelected ? 'ring-2 ring-cyan-400' : 'hover:border-cyan-400/50';

              return (
                <div 
                  key={rack.id}
                  onClick={() => setSelectedRackId(rack.id)}
                  className={`cursor-pointer rounded-lg border p-3 flex flex-col justify-between aspect-square transition-all ${bgClass} ${selectClass}`}
                >
                  <div className="flex justify-between items-start">
                    <span className="font-mono text-sm font-bold text-white">{rack.id}</span>
                    {rack.cooling?.status === 'predictive intervention' && (
                      <span className="text-cyan-400 text-[10px] bg-cyan-400/10 px-1 rounded font-bold">COOL</span>
                    )}
                  </div>
                  <div className="mt-2 text-right font-mono text-xl font-light text-white">
                    {(rack.risk_score * 100).toFixed(0)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Simulation Controls */}
        <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-6 backdrop-blur shadow-2xl flex flex-col">
          <h2 className="text-lg font-bold text-white tracking-wide mb-6">Simulation Controls</h2>
          <CompactBottomBar />
          
          <div className="mt-8">
            <h3 className="text-sm font-bold text-gray-400 tracking-widest uppercase mb-4">Live Event Stream</h3>
            <div className="flex flex-col gap-2 font-mono text-xs">
              {telemetry?.events?.slice(0, 8).map((evt, i) => (
                <div key={i} className="flex gap-3 text-gray-300 border-b border-white/5 pb-2">
                  <span className="text-cyan-400 opacity-70">{evt.time}</span>
                  <span>{evt.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
