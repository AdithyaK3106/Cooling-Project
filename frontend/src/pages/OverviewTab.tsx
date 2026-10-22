import { useState } from 'react';
import { KeyPerformancePanel } from '../features/KeyPerformancePanel';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { useInjectSpike, useToggleRackOverride, useResetSimulation } from '../services/controlApi';
import { SimulationControlBar } from '../components/simulation/SimulationControlBar';
import { XaiExplainerModal } from '../components/xai/XaiExplainerModal';
import { Flame, RefreshCw, Snowflake, Cpu, Sparkles, HardDrive, Wifi, Sliders, Monitor } from 'lucide-react';

export function OverviewTab() {
  const { data: telemetry } = useTelemetry();
  const racks = telemetry?.racks || [];
  const { selectedRackId, setSelectedRackId, uiThemeMode, setUiThemeMode } = useUiStore();
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
    <div 
      style={{ zoom: 1.5 }}
      className="flex h-full w-full flex-col gap-5 p-6 overflow-y-auto bg-[#080A10] text-gray-200 font-sans"
    >
      
      {/* Simulation Physics & Speed Controller Bar at Top */}
      <SimulationControlBar />

      {/* Top Section Header & Mode Switcher Bar */}
      <div className="flex flex-wrap justify-between items-center bg-[#0C1017]/95 p-4 rounded-xl border border-white/[0.08] shadow-2xl backdrop-blur-md gap-4">
        <div>
          <h1 className="text-base font-mono font-bold text-white tracking-wide flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] animate-pulse" />
            Simulated Datacenter Operations Control
          </h1>
          <p className="text-xs text-gray-400 mt-0.5 font-mono">
            25-Rack AI Cluster • Real-Time Telemetry & Predictive Thermal Intelligence
          </p>
        </div>

        {/* UI Mode Selector & Actions */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Mode Switcher Buttons */}
          <div className="flex items-center gap-1 bg-black/50 p-1 rounded-lg border border-white/[0.08] font-mono text-xs">
            <button
              onClick={() => setUiThemeMode('INDUSTRIAL')}
              className={`px-3 py-1.5 rounded-md transition-all font-semibold flex items-center gap-1.5 ${
                uiThemeMode === 'INDUSTRIAL'
                  ? 'bg-zinc-700 text-white font-bold shadow-sm border border-zinc-600'
                  : 'text-gray-400 hover:text-white'
              }`}
              title="Restrained Industrial & Scientific Operations View"
            >
              <Sliders size={13} /> Industrial Ops
            </button>
            <button
              onClick={() => setUiThemeMode('TELEMETRY')}
              className={`px-3 py-1.5 rounded-md transition-all font-semibold flex items-center gap-1.5 ${
                uiThemeMode === 'TELEMETRY'
                  ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/50 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                  : 'text-gray-400 hover:text-white'
              }`}
              title="Full Telemetry & Futuristic Cyberpunk View"
            >
              <Monitor size={13} /> Telemetry FX
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button 
              onClick={() => spikeMutation.mutate()}
              disabled={spikeMutation.isPending}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-red-500/15 text-red-300 border border-red-500/30 text-xs font-mono font-semibold hover:bg-red-500/25 hover:border-red-500/50 transition-all shadow-[0_0_10px_rgba(239,68,68,0.15)] disabled:opacity-50"
            >
              <Flame size={13} /> Inject Spike
            </button>

            <button 
              onClick={() => resetMutation.mutate()}
              disabled={resetMutation.isPending}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-white/[0.05] text-gray-300 border border-white/10 text-xs font-mono font-semibold hover:bg-white/[0.1] hover:text-white transition-all disabled:opacity-50"
            >
              <RefreshCw size={13} /> Reset Sim
            </button>
          </div>
        </div>
      </div>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <KeyPerformancePanel />
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 flex-1 min-h-[520px]">
        
        {/* Left 2 Columns: Datacenter Floor Plan Grid */}
        <div className="lg:col-span-2 rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-5 backdrop-blur-md shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-4 flex-wrap gap-4 border-b border-white/[0.08] pb-3.5">
              <div>
                <h2 className="text-sm font-mono font-bold text-white tracking-wide uppercase">Datacenter Floor Plan (5x5 Rack Array)</h2>
                <p className="text-xs text-gray-400 mt-0.5 font-mono">Real-time Telemetry, Workload Metrics & Predictive Risk</p>
              </div>

              {/* Zone Filter */}
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-mono font-semibold text-gray-400">ZONE:</span>
                {['ALL', 'A', 'B', 'C', 'D', 'E'].map((zone) => (
                  <button
                    key={zone}
                    onClick={() => setZoneFilter(zone)}
                    className={`px-3 py-1 text-xs font-mono rounded-md transition-all ${
                      zoneFilter === zone 
                        ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/50 shadow-[0_0_8px_rgba(6,182,212,0.3)]' 
                        : 'bg-white/[0.04] text-gray-400 border border-white/[0.08] hover:text-white hover:bg-white/[0.08]'
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
                
                let cardColor = 'bg-[#10141D] border-emerald-500/30 text-emerald-300';
                let riskBadgeColor = 'text-emerald-400';
                let riskDot = 'bg-emerald-400 shadow-[0_0_6px_#10b981]';
                
                if (risk >= 0.75) {
                  cardColor = 'bg-red-950/25 border-red-500/50 text-red-300 shadow-[0_0_12px_rgba(239,68,68,0.15)]';
                  riskBadgeColor = 'text-red-400';
                  riskDot = 'bg-red-400 shadow-[0_0_6px_#ef4444]';
                } else if (risk >= 0.55) {
                  cardColor = 'bg-orange-950/25 border-orange-500/50 text-orange-300 shadow-[0_0_12px_rgba(249,115,22,0.15)]';
                  riskBadgeColor = 'text-orange-400';
                  riskDot = 'bg-orange-400 shadow-[0_0_6px_#f97316]';
                } else if (risk >= 0.35) {
                  cardColor = 'bg-amber-950/25 border-amber-500/50 text-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.15)]';
                  riskBadgeColor = 'text-amber-400';
                  riskDot = 'bg-amber-400 shadow-[0_0_6px_#f59e0b]';
                }
                
                const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;
                const isSelected = selectedRackId === rack.id;
                
                const selectClass = isSelected 
                  ? 'border-cyan-400 bg-cyan-950/30 shadow-[0_0_15px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400' 
                  : 'hover:border-white/30 hover:bg-[#141924]';

                const coolingBorder = isCooled ? 'border-l-4 border-l-cyan-400' : '';

                return (
                  <div 
                    key={rack.id}
                    onClick={() => setSelectedRackId(rack.id)}
                    className={`cursor-pointer rounded-xl border p-2.5 flex flex-col justify-between transition-all min-h-[145px] ${cardColor} ${coolingBorder} ${selectClass}`}
                  >
                    {/* Rack Header */}
                    <div className="flex justify-between items-center border-b border-white/[0.08] pb-1.5 mb-1.5">
                      <span className="font-mono text-xs font-bold tracking-wider text-white">{rack.id}</span>
                      {isCooled ? (
                        <span className="text-cyan-300 text-[9px] bg-cyan-950/80 border border-cyan-500/50 px-1.5 py-0.5 rounded font-mono font-semibold flex items-center gap-1 animate-pulse shadow-[0_0_6px_rgba(6,182,212,0.4)]">
                          <Snowflake size={9} /> COOL
                        </span>
                      ) : (
                        <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-white/[0.05] text-gray-400 border border-white/[0.08]">
                          Z-{rack.ai_insights?.zone}
                        </span>
                      )}
                    </div>

                    {/* Detailed Telemetry Rows */}
                    <div className="space-y-1.5 text-[10px] font-mono">
                      {/* CPU */}
                      <div>
                        <div className="flex justify-between items-center text-gray-300">
                          <span>CPU</span>
                          <span className="font-bold text-white">{rack.telemetry?.cpu_util}%</span>
                        </div>
                        <div className="w-full h-1 rounded-full bg-[#181d28] overflow-hidden mt-0.5">
                          <div 
                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300" 
                            style={{ width: `${rack.telemetry?.cpu_util}%` }} 
                          />
                        </div>
                      </div>

                      {/* GPU */}
                      <div>
                        <div className="flex justify-between items-center text-gray-300">
                          <span>GPU</span>
                          <span className="font-bold text-cyan-300">{rack.telemetry?.gpu_util}%</span>
                        </div>
                        <div className="w-full h-1 rounded-full bg-[#181d28] overflow-hidden mt-0.5">
                          <div 
                            className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full transition-all duration-300" 
                            style={{ width: `${rack.telemetry?.gpu_util}%` }} 
                          />
                        </div>
                      </div>

                      {/* RAM & Network Metrics */}
                      <div className="flex justify-between items-center text-[9px] pt-0.5 text-gray-400 border-t border-white/[0.05]">
                        <span>RAM: <strong className="text-blue-300">{rack.telemetry?.mem_util}%</strong></span>
                        <span>IO: <strong className="text-emerald-300">{rack.telemetry?.network_io}M</strong></span>
                      </div>
                    </div>

                    {/* Bottom Dynamic Risk Score Number & XAI Inspector Button */}
                    <div className="flex justify-between items-center border-t border-white/[0.08] pt-1.5 mt-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setXaiModalRack(rack);
                        }}
                        className="text-[9px] font-mono text-cyan-300 hover:text-white bg-cyan-500/15 hover:bg-cyan-500/30 px-1.5 py-0.5 rounded border border-cyan-500/30 transition-all flex items-center gap-1 shadow-[0_0_6px_rgba(6,182,212,0.2)]"
                        title="Explain why the AI is cooling this rack"
                      >
                        <Sparkles size={9} /> XAI
                      </button>
                      <div className="flex items-center gap-1">
                        <span className={`w-1.5 h-1.5 rounded-full ${riskDot}`} />
                        <div className={`font-mono text-xs font-bold ${riskBadgeColor}`}>
                          {riskPct}%
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Grid Footer Legend */}
          <div className="mt-5 flex flex-wrap justify-between items-center text-xs font-mono border-t border-white/[0.08] pt-3 text-gray-400">
            <div className="flex gap-4 items-center flex-wrap">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#10b981]" /> Low (&lt;35%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]" /> Med (35-54%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-orange-400 shadow-[0_0_6px_#f97316]" /> High (55-74%)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_6px_#ef4444]" /> Critical (&ge;75%)</span>
            </div>
            <div className="text-gray-400">Showing <span className="text-white font-bold">{filteredRacks.length}</span> / 25 Racks</div>
          </div>
        </div>

        {/* Right Column: Selected Rack Inspector & Live Audit Stream */}
        <div className="flex flex-col gap-5">
          
          {/* Selected Rack Inspector Panel */}
          {selectedRack ? (
            <div className="rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-5 backdrop-blur-md shadow-2xl flex flex-col gap-3.5">
              <div className="flex justify-between items-center border-b border-white/[0.08] pb-3">
                <div>
                  <h3 className="text-sm font-mono font-bold text-white flex items-center gap-1.5">
                    <Cpu size={16} className="text-cyan-400" />
                    RACK {selectedRack.id}
                  </h3>
                  <span className="text-[11px] font-mono text-gray-400">ZONE {selectedRack.ai_insights?.zone} • SECTOR {selectedRack.id}</span>
                </div>

                <div className="text-right font-mono">
                  <div className="text-[10px] uppercase tracking-wider text-gray-400">AI RISK SCORE</div>
                  <div className={`text-xl font-bold ${selectedRack.risk_score > 0.55 ? 'text-red-400' : selectedRack.risk_score > 0.35 ? 'text-amber-400' : 'text-cyan-300'}`}>
                    {(selectedRack.risk_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Detailed Metrics Grid */}
              <div className="grid grid-cols-2 gap-2.5 text-xs font-mono">
                <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
                  <div className="text-gray-400 text-[10px] uppercase font-bold">CPU UTIL</div>
                  <div className="text-base font-bold text-white mt-0.5">{selectedRack.telemetry?.cpu_util}%</div>
                </div>

                <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
                  <div className="text-gray-400 text-[10px] uppercase font-bold">GPU UTIL</div>
                  <div className="text-base font-bold text-cyan-300 mt-0.5">{selectedRack.telemetry?.gpu_util}%</div>
                </div>

                <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
                  <div className="text-gray-400 text-[10px] uppercase font-bold">RAM MEMORY</div>
                  <div className="text-base font-bold text-blue-300 mt-0.5">{selectedRack.telemetry?.mem_util}%</div>
                </div>

                <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
                  <div className="text-gray-400 flex items-center gap-1 text-[10px] uppercase font-bold">
                    <HardDrive size={11} className="text-amber-400" /> DISK I/O
                  </div>
                  <div className="text-base font-bold text-amber-300 mt-0.5">{selectedRack.telemetry?.disk_io} MB/s</div>
                </div>

                <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
                  <div className="text-gray-400 flex items-center gap-1 text-[10px] uppercase font-bold">
                    <Wifi size={11} className="text-emerald-400" /> NET I/O
                  </div>
                  <div className="text-base font-bold text-emerald-300 mt-0.5">{selectedRack.telemetry?.network_io} MB/s</div>
                </div>

                <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
                  <div className="text-gray-400 text-[10px] uppercase font-bold">POWER DRAW</div>
                  <div className="text-base font-bold text-cyan-400 mt-0.5">{selectedRack.telemetry?.power_draw} W</div>
                </div>
              </div>

              {/* Action Buttons for Selected Rack */}
              <div className="flex gap-2.5 mt-1">
                <button
                  onClick={() => spikeMutation.mutate(selectedRack.id)}
                  className="flex-1 py-2 px-3 bg-red-500/15 border border-red-500/30 text-red-300 rounded-lg text-xs font-mono font-semibold hover:bg-red-500/25 hover:border-red-500/50 transition-all flex items-center justify-center gap-1.5 shadow-[0_0_8px_rgba(239,68,68,0.15)]"
                >
                  <Flame size={14} /> Spike Rack
                </button>

                <button
                  onClick={() => overrideMutation.mutate(selectedRack.id)}
                  className={`flex-1 py-2 px-3 border rounded-lg text-xs font-mono font-semibold transition-all flex items-center justify-center gap-1.5 ${
                    selectedRack.cooling?.override
                      ? 'bg-cyan-500/30 text-cyan-200 border-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.3)]'
                      : 'bg-white/[0.05] border-white/10 text-gray-200 hover:bg-white/[0.1]'
                  }`}
                >
                  <Snowflake size={14} /> {selectedRack.cooling?.override ? 'Release Override' : 'Override Cool'}
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-6 backdrop-blur-md shadow-2xl text-center flex flex-col items-center justify-center min-h-[190px]">
              <div className="p-3 rounded-full bg-white/[0.04] border border-white/[0.08] text-gray-400 mb-2.5">
                <Cpu size={26} />
              </div>
              <div className="text-xs font-mono font-bold text-gray-300 uppercase tracking-wider">NO RACK SELECTED</div>
              <div className="text-[11px] text-gray-400 mt-1 font-mono">Click any rack in the grid to inspect telemetry & AI risk</div>
            </div>
          )}

          {/* Live Events Stream */}
          <div className="rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-5 backdrop-blur-md shadow-2xl flex-1 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-3 border-b border-white/[0.08] pb-2.5">
                <h3 className="text-xs font-mono font-bold text-gray-300 tracking-wider uppercase">Live Audit Stream</h3>
                <span className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#10b981] animate-pulse" />
                  REAL-TIME FEED
                </span>
              </div>

              <div className="flex flex-col gap-2 font-mono text-[11px] max-h-[220px] overflow-y-auto pr-1">
                {telemetry?.events && telemetry.events.length > 0 ? (
                  telemetry.events.map((evt: any, i: number) => {
                    const isWarn = evt.category === 'WARN';
                    const isHealthy = evt.category === 'HEALTHY';
                    
                    return (
                      <div key={i} className="flex items-start gap-2 border-b border-white/[0.04] pb-1.5 text-[11px]">
                        <span className="text-cyan-400 font-bold shrink-0">{evt.time}</span>
                        <span className="text-gray-400 shrink-0">[{evt.source}]</span>
                        <span className={`leading-tight ${isWarn ? 'text-amber-300' : isHealthy ? 'text-emerald-300' : 'text-gray-300'}`}>
                          {evt.message}
                        </span>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-xs py-4 text-center font-mono text-gray-400">No simulation events logged yet</div>
                )}
              </div>
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
