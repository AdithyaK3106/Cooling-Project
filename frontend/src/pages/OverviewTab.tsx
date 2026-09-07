import { useState } from 'react';
import { KeyPerformancePanel } from '../features/KeyPerformancePanel';
import { CompactBottomBar } from '../features/CompactBottomBar';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { useInjectSpike, useToggleRackOverride, useResetSimulation } from '../services/controlApi';
import { Flame, RefreshCw, Snowflake, Cpu, Activity, HardDrive, Wifi, ShieldAlert } from 'lucide-react';

export function OverviewTab() {
  const { data: telemetry } = useTelemetry();
  const racks = telemetry?.racks || [];
  const { selectedRackId, setSelectedRackId } = useUiStore();
  const [zoneFilter, setZoneFilter] = useState<string>('ALL');

  const spikeMutation = useInjectSpike();
  const overrideMutation = useToggleRackOverride();
  const resetMutation = useResetSimulation();

  const selectedRack = racks.find((r: any) => r.id === selectedRackId);

  const filteredRacks = zoneFilter === 'ALL' 
    ? racks 
    : racks.filter((r: any) => r.ai_insights?.zone === zoneFilter);

  return (
    <div className="flex h-full w-full flex-col gap-6 p-6 overflow-y-auto bg-[#07090E] text-white">
      
      {/* Top Section Header & Global Status Bar */}
      <div className="flex flex-wrap justify-between items-center bg-[#0B0E14]/90 p-4 rounded-xl border border-white/10 backdrop-blur shadow-2xl gap-4">
        <div>
          <h1 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse" />
            Simulated Datacenter Mission Control
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            25-Rack High-Density AI Cluster • Real-Time Telemetry & Predictive Thermal Intelligence
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button 
            onClick={() => spikeMutation.mutate()}
            disabled={spikeMutation.isPending}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-600/20 text-red-400 border border-red-500/30 text-xs font-bold hover:bg-red-600/40 transition-all shadow-lg active:scale-95 disabled:opacity-50"
          >
            <Flame size={14} /> Inject Random Spike
          </button>

          <button 
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800 text-gray-300 border border-gray-700 text-xs font-bold hover:bg-gray-700 transition-all shadow-lg active:scale-95 disabled:opacity-50"
          >
            <RefreshCw size={14} /> Reset Simulation
          </button>
        </div>
      </div>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KeyPerformancePanel />
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-[550px]">
        
        {/* Left 2 Columns: Datacenter Floor Plan Grid */}
        <div className="lg:col-span-2 rounded-xl border border-white/10 bg-[#0B0E14]/80 p-6 backdrop-blur shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-6 flex-wrap gap-4">
              <div>
                <h2 className="text-lg font-bold text-white tracking-wide">Datacenter Floor Plan (Interactive 5x5 Grid)</h2>
                <p className="text-xs text-gray-400 mt-0.5">Real-time CPU, GPU, Memory, Disk, Network, & Dynamic Risk Status</p>
              </div>

              {/* Zone Filter */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400 font-mono">ZONE:</span>
                {['ALL', 'A', 'B', 'C', 'D', 'E'].map((zone) => (
                  <button
                    key={zone}
                    onClick={() => setZoneFilter(zone)}
                    className={`px-2.5 py-1 text-xs font-mono rounded transition-all ${
                      zoneFilter === zone 
                        ? 'bg-cyan-500 text-black font-bold shadow-md shadow-cyan-500/20' 
                        : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                    }`}
                  >
                    {zone}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 5x5 Rack Grid with Full Telemetry Breakdown */}
            <div className="grid grid-cols-5 gap-3 content-start">
              {filteredRacks.map((rack: any) => {
                const risk = rack.risk_score || 0;
                const riskPct = Math.round(risk * 100);
                
                // Color status logic based on risk thresholds
                let cardColor = 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300';
                let riskBadgeColor = 'text-emerald-400';
                
                if (risk >= 0.75) {
                  cardColor = 'bg-red-950/60 border-red-500/80 text-red-200 animate-pulse shadow-lg shadow-red-500/20';
                  riskBadgeColor = 'text-red-400';
                } else if (risk >= 0.55) {
                  cardColor = 'bg-orange-950/50 border-orange-500/70 text-orange-200 shadow-md shadow-orange-500/10';
                  riskBadgeColor = 'text-orange-400';
                } else if (risk >= 0.35) {
                  cardColor = 'bg-amber-950/40 border-amber-500/60 text-amber-200';
                  riskBadgeColor = 'text-amber-400';
                }
                
                const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;
                const isSelected = selectedRackId === rack.id;
                
                const selectClass = isSelected 
                  ? 'ring-2 ring-cyan-400 shadow-xl shadow-cyan-500/30 scale-[1.02]' 
                  : 'hover:border-cyan-400/60 hover:scale-[1.01]';

                const coolingBorder = isCooled ? 'border-l-4 border-l-cyan-400' : '';

                return (
                  <div 
                    key={rack.id}
                    onClick={() => setSelectedRackId(rack.id)}
                    className={`cursor-pointer rounded-xl border p-3 flex flex-col justify-between transition-all duration-300 min-h-[145px] ${cardColor} ${coolingBorder} ${selectClass}`}
                  >
                    {/* Rack Header */}
                    <div className="flex justify-between items-center border-b border-white/10 pb-1.5 mb-1.5">
                      <span className="font-mono text-xs font-bold text-white tracking-wider">{rack.id}</span>
                      {isCooled ? (
                        <span className="text-cyan-300 text-[9px] bg-cyan-400/20 border border-cyan-400/40 px-1 py-0.5 rounded font-bold tracking-wider flex items-center gap-0.5">
                          <Snowflake size={9} className="animate-spin" /> COOL
                        </span>
                      ) : (
                        <span className="text-[9px] font-mono text-gray-400">ZONE {rack.ai_insights?.zone}</span>
                      )}
                    </div>

                    {/* Detailed Telemetry Rows (CPU, GPU, RAM, Disk, Net) */}
                    <div className="space-y-1 text-[10px] font-mono">
                      <div className="flex justify-between items-center text-gray-300">
                        <span>CPU</span>
                        <span className="font-bold text-white">{rack.telemetry?.cpu_util}%</span>
                      </div>
                      <div className="w-full h-1 bg-black/50 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-400 transition-all duration-300" style={{ width: `${rack.telemetry?.cpu_util}%` }} />
                      </div>

                      <div className="flex justify-between items-center text-gray-300">
                        <span>GPU</span>
                        <span className="font-bold text-purple-300">{rack.telemetry?.gpu_util}%</span>
                      </div>

                      <div className="flex justify-between items-center text-gray-300">
                        <span>RAM</span>
                        <span className="font-bold text-blue-300">{rack.telemetry?.mem_util}%</span>
                      </div>

                      <div className="flex justify-between items-center text-gray-400 text-[9px] pt-0.5">
                        <span>D:{rack.telemetry?.disk_io}</span>
                        <span>N:{rack.telemetry?.network_io}</span>
                      </div>
                    </div>

                    {/* Bottom Dynamic Risk Score Number (Changes with tick updates!) */}
                    <div className="flex justify-between items-end border-t border-white/10 pt-1.5 mt-1.5">
                      <span className="text-[9px] font-mono text-gray-400">RISK</span>
                      <div className={`font-mono text-base font-bold ${riskBadgeColor} transition-all duration-300`}>
                        {riskPct}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Grid Footer Legend */}
          <div className="mt-6 flex flex-wrap justify-between items-center text-xs font-mono text-gray-400 border-t border-white/10 pt-4">
            <div className="flex gap-4 items-center flex-wrap">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Low (&lt;35%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Med (35-54%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-orange-500" /> High (55-74%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-500" /> Critical (&ge;75%)</span>
            </div>
            <div>Showing {filteredRacks.length} / 25 Racks</div>
          </div>
        </div>

        {/* Right Column: Selected Rack Inspector & Simulation Controls */}
        <div className="flex flex-col gap-6">
          
          {/* Selected Rack Inspector Panel */}
          {selectedRack ? (
            <div className="rounded-xl border border-cyan-500/40 bg-[#0B0E14]/90 p-5 backdrop-blur shadow-2xl flex flex-col gap-4">
              <div className="flex justify-between items-center border-b border-white/10 pb-3">
                <div>
                  <h3 className="text-base font-bold text-white font-mono flex items-center gap-2">
                    <Cpu size={18} className="text-cyan-400" />
                    RACK {selectedRack.id}
                  </h3>
                  <span className="text-xs text-gray-400 font-mono">ZONE {selectedRack.ai_insights?.zone} • SECTOR {selectedRack.id}</span>
                </div>

                <div className="text-right font-mono">
                  <div className="text-xs text-gray-400">RISK SCORE</div>
                  <div className={`text-xl font-bold ${selectedRack.risk_score > 0.55 ? 'text-red-400' : 'text-cyan-400'}`}>
                    {(selectedRack.risk_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Detailed Metrics Grid */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <div className="text-gray-400 flex items-center gap-1"><Cpu size={12}/> CPU UTIL</div>
                  <div className="text-base font-bold text-white mt-0.5">{selectedRack.telemetry?.cpu_util}%</div>
                </div>

                <div className="bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <div className="text-gray-400 flex items-center gap-1"><Activity size={12}/> GPU UTIL</div>
                  <div className="text-base font-bold text-purple-300 mt-0.5">{selectedRack.telemetry?.gpu_util}%</div>
                </div>

                <div className="bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <div className="text-gray-400">RAM MEMORY</div>
                  <div className="text-base font-bold text-blue-300 mt-0.5">{selectedRack.telemetry?.mem_util}%</div>
                </div>

                <div className="bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <div className="text-gray-400 flex items-center gap-1"><HardDrive size={12}/> DISK I/O</div>
                  <div className="text-base font-bold text-amber-300 mt-0.5">{selectedRack.telemetry?.disk_io} MB/s</div>
                </div>

                <div className="bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <div className="text-gray-400 flex items-center gap-1"><Wifi size={12}/> NETWORK I/O</div>
                  <div className="text-base font-bold text-green-300 mt-0.5">{selectedRack.telemetry?.network_io} MB/s</div>
                </div>

                <div className="bg-white/5 p-2.5 rounded-lg border border-white/5">
                  <div className="text-gray-400">POWER DRAW</div>
                  <div className="text-base font-bold text-cyan-400 mt-0.5">{selectedRack.telemetry?.power_draw} W</div>
                </div>
              </div>

              {/* Action Buttons for Selected Rack */}
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => spikeMutation.mutate(selectedRack.id)}
                  className="flex-1 py-2 px-3 bg-red-600/20 border border-red-500/40 text-red-300 rounded-lg text-xs font-bold hover:bg-red-600/40 transition-all flex items-center justify-center gap-1.5"
                >
                  <Flame size={14} /> Spike Rack
                </button>

                <button
                  onClick={() => overrideMutation.mutate(selectedRack.id)}
                  className={`flex-1 py-2 px-3 border rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                    selectedRack.cooling?.override
                      ? 'bg-cyan-500 text-black border-cyan-400'
                      : 'bg-cyan-600/20 border-cyan-500/40 text-cyan-300 hover:bg-cyan-600/40'
                  }`}
                >
                  <Snowflake size={14} /> {selectedRack.cooling?.override ? 'Release Override' : 'Override Cool'}
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur text-center flex flex-col items-center justify-center min-h-[180px]">
              <Cpu size={32} className="text-gray-600 mb-2" />
              <div className="text-sm font-bold text-gray-300">No Rack Selected</div>
              <div className="text-xs text-gray-500 mt-1">Click any rack in the grid to inspect real-time CPU, GPU, RAM, Disk, Network, & AI risk</div>
            </div>
          )}

          {/* Simulation Tuning Controls */}
          <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl flex flex-col gap-4">
            <h2 className="text-sm font-bold text-white tracking-wide uppercase">Simulation Parameters</h2>
            <CompactBottomBar />
          </div>

          {/* Live Events Stream */}
          <div className="rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl flex-1">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-xs font-bold text-gray-400 tracking-widest uppercase">Live Audit Stream</h3>
              <span className="w-2 h-2 rounded-full bg-green-400 animate-ping" />
            </div>

            <div className="flex flex-col gap-2 font-mono text-xs max-h-[220px] overflow-y-auto pr-1">
              {telemetry?.events && telemetry.events.length > 0 ? (
                telemetry.events.map((evt: any, i: number) => (
                  <div key={i} className="flex gap-2 text-gray-300 border-b border-white/5 pb-2 text-[11px]">
                    <span className="text-cyan-400 font-bold">{evt.time}</span>
                    <span className="text-gray-500">[{evt.source}]</span>
                    <span className={evt.category === 'WARN' ? 'text-orange-400' : evt.category === 'HEALTHY' ? 'text-green-400' : 'text-gray-300'}>
                      {evt.message}
                    </span>
                  </div>
                ))
              ) : (
                <div className="text-xs text-gray-500 py-4 text-center">No simulation events logged yet</div>
              )}
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
