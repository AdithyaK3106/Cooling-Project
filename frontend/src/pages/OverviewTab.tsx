import { useState } from 'react';
import { KeyPerformancePanel } from '../features/KeyPerformancePanel';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { useInjectSpike, useToggleRackOverride, useResetSimulation } from '../services/controlApi';
import { SimulationControlBar } from '../components/simulation/SimulationControlBar';
import { XaiExplainerModal } from '../components/xai/XaiExplainerModal';
import { Flame, RefreshCw, Snowflake, Cpu, Sparkles } from 'lucide-react';

export function OverviewTab() {
  const { data: telemetry } = useTelemetry();
  const racks = telemetry?.racks || [];
  const { selectedRackId, setSelectedRackId } = useUiStore();
  const [zoneFilter, setZoneFilter] = useState<string>('ALL');
  const [xaiModalRack, setXaiModalRack] = useState<any>(null);

  const spikeMutation = useInjectSpike();
  const overrideMutation = useToggleRackOverride();
  const resetMutation = useResetSimulation();

  const selectedRack = racks.find((r: any) => r.id === selectedRackId);

  const filteredRacks = zoneFilter === 'ALL' 
    ? racks 
    : racks.filter((r: any) => r.ai_insights?.zone === zoneFilter);

  return (
    <div className="flex h-full w-full flex-col gap-5 p-6 overflow-y-auto bg-[#0e1013] text-neutral-200 font-sans">
      
      {/* Simulation Physics & Speed Controller Bar at Top */}
      <SimulationControlBar />

      {/* Top Section Header & Global Status Bar */}
      <div className="flex flex-wrap justify-between items-center bg-[#15181e] p-4 rounded-md border border-neutral-800 shadow-md gap-4">
        <div>
          <h1 className="text-base font-bold text-neutral-100 tracking-wide flex items-center gap-2 font-sans">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Simulated Datacenter Operations Control
          </h1>
          <p className="text-xs text-neutral-400 mt-0.5 font-mono">
            25-Rack AI Cluster • Real-Time Telemetry & Predictive Thermal Intelligence
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button 
            onClick={() => spikeMutation.mutate()}
            disabled={spikeMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-red-950/70 text-red-300 border border-red-800/80 text-xs font-mono font-semibold hover:bg-red-900/70 transition-colors disabled:opacity-50"
          >
            <Flame size={13} /> Inject Random Spike
          </button>

          <button 
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-neutral-800 text-neutral-300 border border-neutral-700 text-xs font-mono font-semibold hover:bg-neutral-700 transition-colors disabled:opacity-50"
          >
            <RefreshCw size={13} /> Reset Simulation
          </button>
        </div>
      </div>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <KeyPerformancePanel />
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 flex-1 min-h-[520px]">
        
        {/* Left 2 Columns: Datacenter Floor Plan Grid */}
        <div className="lg:col-span-2 rounded-md border border-neutral-800 bg-[#15181e] p-5 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-5 flex-wrap gap-4 border-b border-neutral-800/80 pb-3">
              <div>
                <h2 className="text-sm font-bold text-neutral-100 tracking-wide font-sans uppercase">Datacenter Floor Plan (5x5 Rack Array)</h2>
                <p className="text-xs text-neutral-400 mt-0.5 font-mono">Real-time Telemetry, Workload Metrics & Predictive Risk</p>
              </div>

              {/* Zone Filter */}
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-neutral-400 font-mono font-semibold">ZONE:</span>
                {['ALL', 'A', 'B', 'C', 'D', 'E'].map((zone) => (
                  <button
                    key={zone}
                    onClick={() => setZoneFilter(zone)}
                    className={`px-2.5 py-1 text-xs font-mono rounded-sm transition-colors ${
                      zoneFilter === zone 
                        ? 'bg-zinc-700 text-neutral-100 font-bold border border-zinc-600' 
                        : 'bg-neutral-900 text-neutral-400 border border-neutral-800 hover:text-neutral-200'
                    }`}
                  >
                    {zone}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 5x5 Rack Grid with Full Telemetry Breakdown */}
            <div className="grid grid-cols-5 gap-2.5 content-start">
              {filteredRacks.map((rack: any) => {
                const risk = rack.risk_score || 0;
                const riskPct = Math.round(risk * 100);
                
                // Industrial muted state colors
                let cardColor = 'bg-[#181b20] border-emerald-800/60 text-emerald-300';
                let riskBadgeColor = 'text-emerald-400';
                
                if (risk >= 0.75) {
                  cardColor = 'bg-red-950/40 border-red-800/80 text-red-300';
                  riskBadgeColor = 'text-red-400';
                } else if (risk >= 0.55) {
                  cardColor = 'bg-orange-950/40 border-orange-800/80 text-orange-300';
                  riskBadgeColor = 'text-orange-400';
                } else if (risk >= 0.35) {
                  cardColor = 'bg-amber-950/40 border-amber-800/80 text-amber-300';
                  riskBadgeColor = 'text-amber-400';
                }
                
                const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;
                const isSelected = selectedRackId === rack.id;
                
                const selectClass = isSelected 
                  ? 'border-zinc-300 bg-[#222731]' 
                  : 'hover:border-neutral-600';

                const coolingBorder = isCooled ? 'border-l-2 border-l-slate-400' : '';

                return (
                  <div 
                    key={rack.id}
                    onClick={() => setSelectedRackId(rack.id)}
                    className={`cursor-pointer rounded-md border p-2.5 flex flex-col justify-between transition-colors min-h-[140px] ${cardColor} ${coolingBorder} ${selectClass}`}
                  >
                    {/* Rack Header */}
                    <div className="flex justify-between items-center border-b border-neutral-800 pb-1 mb-1">
                      <span className="font-mono text-xs font-bold text-neutral-100 tracking-wider">{rack.id}</span>
                      {isCooled ? (
                        <span className="text-slate-200 text-[9px] bg-slate-800 border border-slate-700 px-1 py-0.5 rounded font-mono font-semibold flex items-center gap-0.5">
                          <Snowflake size={8} /> COOL
                        </span>
                      ) : (
                        <span className="text-[9px] font-mono text-neutral-400">ZONE {rack.ai_insights?.zone}</span>
                      )}
                    </div>

                    {/* Detailed Telemetry Rows */}
                    <div className="space-y-1 text-[10px] font-mono">
                      <div className="flex justify-between items-center text-neutral-300">
                        <span>CPU</span>
                        <span className="font-bold text-neutral-100">{rack.telemetry?.cpu_util}%</span>
                      </div>
                      <div className="w-full h-1 bg-neutral-900 rounded-sm overflow-hidden">
                        <div className="h-full bg-zinc-400 transition-all duration-300" style={{ width: `${rack.telemetry?.cpu_util}%` }} />
                      </div>

                      <div className="flex justify-between items-center text-neutral-300">
                        <span>GPU</span>
                        <span className="font-bold text-neutral-200">{rack.telemetry?.gpu_util}%</span>
                      </div>

                      <div className="flex justify-between items-center text-neutral-300">
                        <span>RAM</span>
                        <span className="font-bold text-neutral-200">{rack.telemetry?.mem_util}%</span>
                      </div>

                      <div className="flex justify-between items-center text-neutral-400 text-[9px] pt-0.5">
                        <span>D:{rack.telemetry?.disk_io}</span>
                        <span>N:{rack.telemetry?.network_io}</span>
                      </div>
                    </div>

                    {/* Bottom Dynamic Risk Score Number & XAI Inspector Button */}
                    <div className="flex justify-between items-center border-t border-neutral-800 pt-1 mt-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setXaiModalRack(rack);
                        }}
                        className="text-[9px] font-mono text-neutral-300 hover:text-white bg-neutral-800 hover:bg-neutral-700 px-1.5 py-0.5 rounded border border-neutral-700 transition-colors flex items-center gap-1"
                        title="Explain why the AI is cooling this rack"
                      >
                        <Sparkles size={9} /> XAI
                      </button>
                      <div className={`font-mono text-sm font-bold ${riskBadgeColor}`}>
                        {riskPct}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Grid Footer Legend */}
          <div className="mt-5 flex flex-wrap justify-between items-center text-xs font-mono text-neutral-400 border-t border-neutral-800 pt-3">
            <div className="flex gap-4 items-center flex-wrap">
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500" /> Low (&lt;35%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500" /> Med (35-54%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-orange-500" /> High (55-74%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500" /> Critical (&ge;75%)</span>
            </div>
            <div>Showing {filteredRacks.length} / 25 Racks</div>
          </div>
        </div>

        {/* Right Column: Selected Rack Inspector & Live Audit Stream */}
        <div className="flex flex-col gap-5">
          
          {/* Selected Rack Inspector Panel */}
          {selectedRack ? (
            <div className="rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md flex flex-col gap-3.5">
              <div className="flex justify-between items-center border-b border-neutral-800 pb-2.5">
                <div>
                  <h3 className="text-sm font-bold text-neutral-100 font-mono flex items-center gap-1.5">
                    <Cpu size={16} className="text-neutral-300" />
                    RACK {selectedRack.id}
                  </h3>
                  <span className="text-[11px] text-neutral-400 font-mono">ZONE {selectedRack.ai_insights?.zone} • SECTOR {selectedRack.id}</span>
                </div>

                <div className="text-right font-mono">
                  <div className="text-[10px] text-neutral-400 uppercase tracking-wider">RISK SCORE</div>
                  <div className={`text-lg font-bold ${selectedRack.risk_score > 0.55 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {(selectedRack.risk_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Detailed Metrics Grid */}
              <div className="grid grid-cols-2 gap-2.5 text-xs font-mono">
                <div className="bg-[#181b20] p-2 rounded border border-neutral-800">
                  <div className="text-neutral-400 flex items-center gap-1 text-[10px] uppercase">CPU UTIL</div>
                  <div className="text-sm font-bold text-neutral-100 mt-0.5">{selectedRack.telemetry?.cpu_util}%</div>
                </div>

                <div className="bg-[#181b20] p-2 rounded border border-neutral-800">
                  <div className="text-neutral-400 flex items-center gap-1 text-[10px] uppercase">GPU UTIL</div>
                  <div className="text-sm font-bold text-neutral-100 mt-0.5">{selectedRack.telemetry?.gpu_util}%</div>
                </div>

                <div className="bg-[#181b20] p-2 rounded border border-neutral-800">
                  <div className="text-neutral-400 text-[10px] uppercase">RAM MEMORY</div>
                  <div className="text-sm font-bold text-neutral-100 mt-0.5">{selectedRack.telemetry?.mem_util}%</div>
                </div>

                <div className="bg-[#181b20] p-2 rounded border border-neutral-800">
                  <div className="text-neutral-400 text-[10px] uppercase flex items-center gap-1">DISK I/O</div>
                  <div className="text-sm font-bold text-neutral-100 mt-0.5">{selectedRack.telemetry?.disk_io} MB/s</div>
                </div>

                <div className="bg-[#181b20] p-2 rounded border border-neutral-800">
                  <div className="text-neutral-400 text-[10px] uppercase flex items-center gap-1">NETWORK I/O</div>
                  <div className="text-sm font-bold text-neutral-100 mt-0.5">{selectedRack.telemetry?.network_io} MB/s</div>
                </div>

                <div className="bg-[#181b20] p-2 rounded border border-neutral-800">
                  <div className="text-neutral-400 text-[10px] uppercase">POWER DRAW</div>
                  <div className="text-sm font-bold text-neutral-100 mt-0.5">{selectedRack.telemetry?.power_draw} W</div>
                </div>
              </div>

              {/* Action Buttons for Selected Rack */}
              <div className="flex gap-2 mt-1">
                <button
                  onClick={() => spikeMutation.mutate(selectedRack.id)}
                  className="flex-1 py-1.5 px-2.5 bg-red-950/70 border border-red-800 text-red-300 rounded text-xs font-mono font-semibold hover:bg-red-900/70 transition-colors flex items-center justify-center gap-1.5"
                >
                  <Flame size={13} /> Spike Rack
                </button>

                <button
                  onClick={() => overrideMutation.mutate(selectedRack.id)}
                  className={`flex-1 py-1.5 px-2.5 border rounded text-xs font-mono font-semibold transition-colors flex items-center justify-center gap-1.5 ${
                    selectedRack.cooling?.override
                      ? 'bg-zinc-700 text-white border-zinc-500'
                      : 'bg-neutral-800 border-neutral-700 text-neutral-200 hover:bg-neutral-700'
                  }`}
                >
                  <Snowflake size={13} /> {selectedRack.cooling?.override ? 'Release Override' : 'Override Cool'}
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-md border border-neutral-800 bg-[#15181e] p-4 text-center flex flex-col items-center justify-center min-h-[160px]">
              <Cpu size={28} className="text-neutral-600 mb-2" />
              <div className="text-xs font-mono font-bold text-neutral-300">NO RACK SELECTED</div>
              <div className="text-[11px] text-neutral-400 mt-1">Click any rack in the grid to inspect telemetry & AI risk</div>
            </div>
          )}

          {/* Live Events Stream */}
          <div className="rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md flex-1">
            <div className="flex justify-between items-center mb-2.5 border-b border-neutral-800/80 pb-2">
              <h3 className="text-xs font-bold text-neutral-400 tracking-widest uppercase font-mono">Live Audit Stream</h3>
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
            </div>

            <div className="flex flex-col gap-1.5 font-mono text-[11px] max-h-[220px] overflow-y-auto pr-1">
              {telemetry?.events && telemetry.events.length > 0 ? (
                telemetry.events.map((evt: any, i: number) => (
                  <div key={i} className="flex gap-2 text-neutral-300 border-b border-neutral-800/40 pb-1.5 text-[11px]">
                    <span className="text-neutral-400 font-bold">{evt.time}</span>
                    <span className="text-neutral-500">[{evt.source}]</span>
                    <span className={evt.category === 'WARN' ? 'text-amber-400' : evt.category === 'HEALTHY' ? 'text-emerald-400' : 'text-neutral-300'}>
                      {evt.message}
                    </span>
                  </div>
                ))
              ) : (
                <div className="text-xs text-neutral-500 py-3 text-center font-mono">No simulation events logged yet</div>
              )}
            </div>
          </div>

        </div>

      </div>

      {/* XAI Explainer Inspector Modal */}
      <XaiExplainerModal
        rack={xaiModalRack}
        isOpen={!!xaiModalRack}
        onClose={() => setXaiModalRack(null)}
      />
    </div>
  );
}
