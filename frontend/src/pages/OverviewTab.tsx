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

  const isTelemetry = uiThemeMode === 'TELEMETRY';

  const spikeMutation = useInjectSpike();
  const overrideMutation = useToggleRackOverride();
  const resetMutation = useResetSimulation();

  const selectedRack = racks.find((r: any) => r.id === selectedRackId);

  const filteredRacks = zoneFilter === 'ALL' 
    ? racks 
    : racks.filter((r: any) => r.ai_insights?.zone === zoneFilter);

  return (
    <div className={`flex h-full w-full flex-col gap-5 p-6 overflow-y-auto ${isTelemetry ? 'bg-[#0B0E14] text-white' : 'bg-[#0e1013] text-neutral-200'} font-sans`}>
      
      {/* Simulation Physics & Speed Controller Bar at Top */}
      <SimulationControlBar />

      {/* Top Section Header & Mode Switcher Bar */}
      <div className={isTelemetry 
        ? "flex flex-wrap justify-between items-center bg-[#0B0E14]/80 p-4 rounded-xl border border-white/10 shadow-2xl backdrop-blur gap-4" 
        : "flex flex-wrap justify-between items-center bg-[#15181e] p-4 rounded-md border border-neutral-800 shadow-md gap-4"
      }>
        <div>
          <h1 className={isTelemetry ? "text-base font-bold text-white tracking-wide flex items-center gap-2 font-mono" : "text-base font-bold text-neutral-100 tracking-wide flex items-center gap-2 font-sans"}>
            <span className={isTelemetry ? "w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] animate-pulse" : "w-2 h-2 rounded-full bg-emerald-500"} />
            Simulated Datacenter Operations Control
          </h1>
          <p className={isTelemetry ? "text-xs text-gray-400 mt-0.5 font-mono" : "text-xs text-neutral-400 mt-0.5 font-mono"}>
            25-Rack AI Cluster • Real-Time Telemetry & Predictive Thermal Intelligence
          </p>
        </div>

        {/* UI Mode Selector & Actions */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Mode Switcher Buttons */}
          <div className="flex items-center gap-1 bg-black/60 p-1 rounded-md border border-white/10 font-mono text-xs">
            <button
              onClick={() => setUiThemeMode('INDUSTRIAL')}
              className={`px-3 py-1.5 rounded transition-all font-semibold flex items-center gap-1.5 ${
                uiThemeMode === 'INDUSTRIAL'
                  ? 'bg-zinc-700 text-white font-bold shadow-sm'
                  : 'text-gray-400 hover:text-white'
              }`}
              title="Restrained Industrial & Scientific Operations View"
            >
              <Sliders size={13} /> Industrial Ops
            </button>
            <button
              onClick={() => setUiThemeMode('TELEMETRY')}
              className={`px-3 py-1.5 rounded transition-all font-semibold flex items-center gap-1.5 ${
                uiThemeMode === 'TELEMETRY'
                  ? 'bg-cyan-500 text-black font-bold shadow-[0_0_12px_#06b6d4]'
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
              className={isTelemetry 
                ? "flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-600/20 text-red-300 border border-red-500/40 text-xs font-bold hover:bg-red-600/40 transition-colors disabled:opacity-50"
                : "flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-red-950/70 text-red-300 border border-red-800/80 text-xs font-mono font-semibold hover:bg-red-900/70 transition-colors disabled:opacity-50"
              }
            >
              <Flame size={13} /> Inject Spike
            </button>

            <button 
              onClick={() => resetMutation.mutate()}
              disabled={resetMutation.isPending}
              className={isTelemetry
                ? "flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 text-gray-300 border border-white/10 text-xs font-bold hover:bg-white/20 transition-colors disabled:opacity-50"
                : "flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-neutral-800 text-neutral-300 border border-neutral-700 text-xs font-mono font-semibold hover:bg-neutral-700 transition-colors disabled:opacity-50"
              }
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
        <div className={isTelemetry 
          ? "lg:col-span-2 rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl flex flex-col justify-between" 
          : "lg:col-span-2 rounded-md border border-neutral-800 bg-[#15181e] p-5 shadow-md flex flex-col justify-between"
        }>
          <div>
            <div className={`flex justify-between items-center mb-5 flex-wrap gap-4 border-b pb-3 ${isTelemetry ? 'border-white/10' : 'border-neutral-800/80'}`}>
              <div>
                <h2 className={isTelemetry ? "text-sm font-bold text-white tracking-wide font-mono uppercase" : "text-sm font-bold text-neutral-100 tracking-wide font-sans uppercase"}>Datacenter Floor Plan (5x5 Rack Array)</h2>
                <p className={isTelemetry ? "text-xs text-gray-400 mt-0.5 font-mono" : "text-xs text-neutral-400 mt-0.5 font-mono"}>Real-time Telemetry, Workload Metrics & Predictive Risk</p>
              </div>

              {/* Zone Filter */}
              <div className="flex items-center gap-1.5">
                <span className={`text-[11px] font-mono font-semibold ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>ZONE:</span>
                {['ALL', 'A', 'B', 'C', 'D', 'E'].map((zone) => (
                  <button
                    key={zone}
                    onClick={() => setZoneFilter(zone)}
                    className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                      zoneFilter === zone 
                        ? (isTelemetry ? 'bg-cyan-500 text-black font-bold shadow-[0_0_8px_#06b6d4]' : 'bg-zinc-700 text-neutral-100 font-bold border border-zinc-600') 
                        : (isTelemetry ? 'bg-white/5 text-gray-400 border border-white/10 hover:text-white' : 'bg-neutral-900 text-neutral-400 border border-neutral-800 hover:text-neutral-200')
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
                
                let cardColor = isTelemetry 
                  ? 'bg-white/5 border-cyan-500/30 text-cyan-300' 
                  : 'bg-[#181b20] border-emerald-800/60 text-emerald-300';
                
                let riskBadgeColor = isTelemetry 
                  ? 'text-cyan-400 drop-shadow-[0_0_6px_rgba(34,211,238,0.6)]' 
                  : 'text-emerald-400';
                
                if (risk >= 0.75) {
                  cardColor = isTelemetry 
                    ? 'bg-red-950/30 border-red-500/60 text-red-300 shadow-[0_0_12px_rgba(239,68,68,0.2)]' 
                    : 'bg-red-950/40 border-red-800/80 text-red-300';
                  riskBadgeColor = isTelemetry 
                    ? 'text-red-400 drop-shadow-[0_0_6px_rgba(239,68,68,0.6)]' 
                    : 'text-red-400';
                } else if (risk >= 0.55) {
                  cardColor = isTelemetry 
                    ? 'bg-orange-950/30 border-orange-500/60 text-orange-300 shadow-[0_0_12px_rgba(249,115,22,0.2)]' 
                    : 'bg-orange-950/40 border-orange-800/80 text-orange-300';
                  riskBadgeColor = isTelemetry 
                    ? 'text-orange-400 drop-shadow-[0_0_6px_rgba(249,115,22,0.6)]' 
                    : 'text-orange-400';
                } else if (risk >= 0.35) {
                  cardColor = isTelemetry 
                    ? 'bg-amber-950/30 border-amber-500/60 text-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.2)]' 
                    : 'bg-amber-950/40 border-amber-800/80 text-amber-300';
                  riskBadgeColor = isTelemetry 
                    ? 'text-amber-400 drop-shadow-[0_0_6px_rgba(245,158,11,0.6)]' 
                    : 'text-amber-400';
                }
                
                const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;
                const isSelected = selectedRackId === rack.id;
                
                const selectClass = isSelected 
                  ? (isTelemetry ? 'border-cyan-400 bg-cyan-950/40 shadow-[0_0_15px_#06b6d4]' : 'border-zinc-300 bg-[#222731]') 
                  : (isTelemetry ? 'hover:border-cyan-500/60' : 'hover:border-neutral-600');

                const coolingBorder = isCooled ? (isTelemetry ? 'border-l-4 border-l-cyan-400' : 'border-l-2 border-l-slate-400') : '';

                return (
                  <div 
                    key={rack.id}
                    onClick={() => setSelectedRackId(rack.id)}
                    className={`cursor-pointer rounded-xl border p-2.5 flex flex-col justify-between transition-all min-h-[140px] ${cardColor} ${coolingBorder} ${selectClass}`}
                  >
                    {/* Rack Header */}
                    <div className={`flex justify-between items-center border-b pb-1 mb-1 ${isTelemetry ? 'border-white/10' : 'border-neutral-800'}`}>
                      <span className={`font-mono text-xs font-bold tracking-wider ${isTelemetry ? 'text-white' : 'text-neutral-100'}`}>{rack.id}</span>
                      {isCooled ? (
                        <span className={isTelemetry 
                          ? "text-cyan-300 text-[9px] bg-cyan-950/80 border border-cyan-500/50 px-1 py-0.5 rounded font-mono font-semibold flex items-center gap-0.5 animate-pulse shadow-[0_0_6px_#06b6d4]" 
                          : "text-slate-200 text-[9px] bg-slate-800 border border-slate-700 px-1 py-0.5 rounded font-mono font-semibold flex items-center gap-0.5"
                        }>
                          <Snowflake size={8} /> COOL
                        </span>
                      ) : (
                        <span className={`text-[9px] font-mono ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>ZONE {rack.ai_insights?.zone}</span>
                      )}
                    </div>

                    {/* Detailed Telemetry Rows */}
                    <div className="space-y-1 text-[10px] font-mono">
                      <div className={`flex justify-between items-center ${isTelemetry ? 'text-gray-300' : 'text-neutral-300'}`}>
                        <span>CPU</span>
                        <span className={`font-bold ${isTelemetry ? 'text-cyan-400' : 'text-neutral-100'}`}>{rack.telemetry?.cpu_util}%</span>
                      </div>
                      <div className={`w-full h-1 rounded-sm overflow-hidden ${isTelemetry ? 'bg-white/10' : 'bg-neutral-900'}`}>
                        <div className={`h-full transition-all duration-300 ${isTelemetry ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-[0_0_6px_#06b6d4]' : 'bg-zinc-400'}`} style={{ width: `${rack.telemetry?.cpu_util}%` }} />
                      </div>

                      <div className={`flex justify-between items-center ${isTelemetry ? 'text-gray-300' : 'text-neutral-300'}`}>
                        <span>GPU</span>
                        <span className={`font-bold ${isTelemetry ? 'text-cyan-400' : 'text-neutral-200'}`}>{rack.telemetry?.gpu_util}%</span>
                      </div>

                      <div className={`flex justify-between items-center ${isTelemetry ? 'text-gray-300' : 'text-neutral-300'}`}>
                        <span>RAM</span>
                        <span className={`font-bold ${isTelemetry ? 'text-blue-300' : 'text-neutral-200'}`}>{rack.telemetry?.mem_util}%</span>
                      </div>

                      <div className={`flex justify-between items-center text-[9px] pt-0.5 ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>
                        <span className={isTelemetry ? 'text-amber-300 font-semibold' : ''}>D:{rack.telemetry?.disk_io}</span>
                        <span className={isTelemetry ? 'text-green-300 font-semibold' : ''}>N:{rack.telemetry?.network_io}</span>
                      </div>
                    </div>

                    {/* Bottom Dynamic Risk Score Number & XAI Inspector Button */}
                    <div className={`flex justify-between items-center border-t pt-1 mt-1 ${isTelemetry ? 'border-white/10' : 'border-neutral-800'}`}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setXaiModalRack(rack);
                        }}
                        className={isTelemetry
                          ? "text-[9px] font-mono text-cyan-300 hover:text-white bg-cyan-500/20 hover:bg-cyan-500/40 px-1.5 py-0.5 rounded border border-cyan-500/40 transition-colors flex items-center gap-1 shadow-[0_0_6px_rgba(6,182,212,0.3)]"
                          : "text-[9px] font-mono text-neutral-300 hover:text-white bg-neutral-800 hover:bg-neutral-700 px-1.5 py-0.5 rounded border border-neutral-700 transition-colors flex items-center gap-1"
                        }
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
          <div className={`mt-5 flex flex-wrap justify-between items-center text-xs font-mono border-t pt-3 ${isTelemetry ? 'text-gray-400 border-white/10' : 'text-neutral-400 border-neutral-800'}`}>
            <div className="flex gap-4 items-center flex-wrap">
              <span className="flex items-center gap-1.5"><span className={isTelemetry ? "w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_6px_#06b6d4]" : "w-2 h-2 rounded-full bg-emerald-500"} /> Low (&lt;35%)</span>
              <span className="flex items-center gap-1.5"><span className={isTelemetry ? "w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]" : "w-2 h-2 rounded-full bg-amber-500"} /> Med (35-54%)</span>
              <span className="flex items-center gap-1.5"><span className={isTelemetry ? "w-2.5 h-2.5 rounded-full bg-orange-400 shadow-[0_0_6px_#f97316]" : "w-2 h-2 rounded-full bg-orange-500"} /> High (55-74%)</span>
              <span className="flex items-center gap-1.5"><span className={isTelemetry ? "w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_6px_#ef4444]" : "w-2 h-2 rounded-full bg-red-500"} /> Critical (&ge;75%)</span>
            </div>
            <div>Showing {filteredRacks.length} / 25 Racks</div>
          </div>
        </div>

        {/* Right Column: Selected Rack Inspector & Live Audit Stream */}
        <div className="flex flex-col gap-5">
          
          {/* Selected Rack Inspector Panel */}
          {selectedRack ? (
            <div className={isTelemetry 
              ? "rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl flex flex-col gap-3.5" 
              : "rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md flex flex-col gap-3.5"
            }>
              <div className={`flex justify-between items-center border-b pb-2.5 ${isTelemetry ? 'border-white/10' : 'border-neutral-800'}`}>
                <div>
                  <h3 className={isTelemetry 
                    ? "text-sm font-bold text-cyan-400 font-mono flex items-center gap-1.5 drop-shadow-[0_0_8px_rgba(34,211,238,0.4)]" 
                    : "text-sm font-bold text-neutral-100 font-mono flex items-center gap-1.5"
                  }>
                    <Cpu size={16} className={isTelemetry ? "text-cyan-400" : "text-neutral-300"} />
                    RACK {selectedRack.id}
                  </h3>
                  <span className={`text-[11px] font-mono ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>ZONE {selectedRack.ai_insights?.zone} • SECTOR {selectedRack.id}</span>
                </div>

                <div className="text-right font-mono">
                  <div className={`text-[10px] uppercase tracking-wider ${isTelemetry ? 'text-gray-400' : 'text-neutral-400'}`}>RISK SCORE</div>
                  <div className={`text-lg font-bold ${selectedRack.risk_score > 0.55 ? 'text-red-400' : (isTelemetry ? 'text-cyan-400' : 'text-emerald-400')}`}>
                    {(selectedRack.risk_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Detailed Metrics Grid */}
              <div className="grid grid-cols-2 gap-2.5 text-xs font-mono">
                <div className={isTelemetry ? "bg-white/5 p-2.5 rounded-lg border border-white/5" : "bg-[#181b20] p-2 rounded border border-neutral-800"}>
                  <div className={isTelemetry ? "text-gray-400 text-[10px] uppercase font-bold" : "text-neutral-400 text-[10px] uppercase flex items-center gap-1"}>CPU UTIL</div>
                  <div className={isTelemetry ? "text-base font-bold text-cyan-400 mt-0.5" : "text-sm font-bold text-neutral-100 mt-0.5"}>{selectedRack.telemetry?.cpu_util}%</div>
                </div>

                <div className={isTelemetry ? "bg-white/5 p-2.5 rounded-lg border border-white/5" : "bg-[#181b20] p-2 rounded border border-neutral-800"}>
                  <div className={isTelemetry ? "text-gray-400 text-[10px] uppercase font-bold" : "text-neutral-400 text-[10px] uppercase flex items-center gap-1"}>GPU UTIL</div>
                  <div className={isTelemetry ? "text-base font-bold text-cyan-400 mt-0.5" : "text-sm font-bold text-neutral-100 mt-0.5"}>{selectedRack.telemetry?.gpu_util}%</div>
                </div>

                <div className={isTelemetry ? "bg-white/5 p-2.5 rounded-lg border border-white/5" : "bg-[#181b20] p-2 rounded border border-neutral-800"}>
                  <div className={isTelemetry ? "text-gray-400 text-[10px] uppercase font-bold" : "text-neutral-400 text-[10px] uppercase"}>RAM MEMORY</div>
                  <div className={isTelemetry ? "text-base font-bold text-blue-300 mt-0.5" : "text-sm font-bold text-neutral-100 mt-0.5"}>{selectedRack.telemetry?.mem_util}%</div>
                </div>

                <div className={isTelemetry ? "bg-white/5 p-2.5 rounded-lg border border-white/5" : "bg-[#181b20] p-2 rounded border border-neutral-800"}>
                  <div className={isTelemetry ? "text-gray-400 flex items-center gap-1 text-[10px] uppercase font-bold" : "text-neutral-400 text-[10px] uppercase flex items-center gap-1"}>
                    {isTelemetry && <HardDrive size={12} />} DISK I/O
                  </div>
                  <div className={isTelemetry ? "text-base font-bold text-amber-300 mt-0.5" : "text-sm font-bold text-neutral-100 mt-0.5"}>{selectedRack.telemetry?.disk_io} MB/s</div>
                </div>

                <div className={isTelemetry ? "bg-white/5 p-2.5 rounded-lg border border-white/5" : "bg-[#181b20] p-2 rounded border border-neutral-800"}>
                  <div className={isTelemetry ? "text-gray-400 flex items-center gap-1 text-[10px] uppercase font-bold" : "text-neutral-400 text-[10px] uppercase flex items-center gap-1"}>
                    {isTelemetry && <Wifi size={12} />} NETWORK I/O
                  </div>
                  <div className={isTelemetry ? "text-base font-bold text-green-300 mt-0.5" : "text-sm font-bold text-neutral-100 mt-0.5"}>{selectedRack.telemetry?.network_io} MB/s</div>
                </div>

                <div className={isTelemetry ? "bg-white/5 p-2.5 rounded-lg border border-white/5" : "bg-[#181b20] p-2 rounded border border-neutral-800"}>
                  <div className={isTelemetry ? "text-gray-400 text-[10px] uppercase font-bold" : "text-neutral-400 text-[10px] uppercase"}>POWER DRAW</div>
                  <div className={isTelemetry ? "text-base font-bold text-cyan-400 mt-0.5" : "text-sm font-bold text-neutral-100 mt-0.5"}>{selectedRack.telemetry?.power_draw} W</div>
                </div>
              </div>

              {/* Action Buttons for Selected Rack */}
              <div className="flex gap-2 mt-1">
                <button
                  onClick={() => spikeMutation.mutate(selectedRack.id)}
                  className={isTelemetry
                    ? "flex-1 py-2 px-3 bg-red-600/20 border border-red-500/40 text-red-300 rounded-lg text-xs font-bold hover:bg-red-600/40 transition-all flex items-center justify-center gap-1.5"
                    : "flex-1 py-1.5 px-2.5 bg-red-950/70 border border-red-800 text-red-300 rounded text-xs font-mono font-semibold hover:bg-red-900/70 transition-colors flex items-center justify-center gap-1.5"
                  }
                >
                  <Flame size={14} /> Spike Rack
                </button>

                <button
                  onClick={() => overrideMutation.mutate(selectedRack.id)}
                  className={isTelemetry
                    ? `flex-1 py-2 px-3 border rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                        selectedRack.cooling?.override
                          ? 'bg-cyan-500 text-black border-cyan-400 shadow-[0_0_10px_#06b6d4]'
                          : 'bg-cyan-600/20 border-cyan-500/40 text-cyan-300 hover:bg-cyan-600/40'
                      }`
                    : `flex-1 py-1.5 px-2.5 border rounded text-xs font-mono font-semibold transition-colors flex items-center justify-center gap-1.5 ${
                        selectedRack.cooling?.override
                          ? 'bg-zinc-700 text-white border-zinc-500'
                          : 'bg-neutral-800 border-neutral-700 text-neutral-200 hover:bg-neutral-700'
                      }`
                  }
                >
                  <Snowflake size={14} /> {selectedRack.cooling?.override ? 'Release Override' : 'Override Cool'}
                </button>
              </div>
            </div>
          ) : (
            <div className={isTelemetry 
              ? "rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl text-center flex flex-col items-center justify-center min-h-[180px]" 
              : "rounded-md border border-neutral-800 bg-[#15181e] p-4 text-center flex flex-col items-center justify-center min-h-[160px]"
            }>
              <Cpu size={28} className={isTelemetry ? "text-cyan-500/50 mb-2" : "text-neutral-600 mb-2"} />
              <div className={isTelemetry ? "text-sm font-bold text-gray-300 font-mono" : "text-xs font-mono font-bold text-neutral-300"}>NO RACK SELECTED</div>
              <div className={isTelemetry ? "text-xs text-gray-500 mt-1 font-mono" : "text-[11px] text-neutral-400 mt-1 font-mono"}>Click any rack in the grid to inspect telemetry & AI risk</div>
            </div>
          )}

          {/* Live Events Stream */}
          <div className={isTelemetry 
            ? "rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur shadow-2xl flex-1" 
            : "rounded-md border border-neutral-800 bg-[#15181e] p-4 shadow-md flex-1"
          }>
            <div className={`flex justify-between items-center mb-2.5 border-b pb-2 ${isTelemetry ? 'border-white/10' : 'border-neutral-800/80'}`}>
              <h3 className={isTelemetry ? "text-xs font-bold text-gray-400 tracking-widest uppercase font-mono" : "text-xs font-bold text-neutral-400 tracking-widest uppercase font-mono"}>Live Audit Stream</h3>
              {isTelemetry ? (
                <span className="w-2.5 h-2.5 rounded-full bg-green-400 animate-ping shadow-[0_0_8px_#22c55e]" />
              ) : (
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
              )}
            </div>

            <div className="flex flex-col gap-1.5 font-mono text-[11px] max-h-[220px] overflow-y-auto pr-1">
              {telemetry?.events && telemetry.events.length > 0 ? (
                telemetry.events.map((evt: any, i: number) => (
                  <div key={i} className={`flex gap-2 border-b pb-1.5 text-[11px] ${isTelemetry ? 'text-gray-300 border-white/5' : 'text-neutral-300 border-neutral-800/40'}`}>
                    <span className={isTelemetry ? 'text-cyan-400 font-bold' : 'text-neutral-400 font-bold'}>{evt.time}</span>
                    <span className={isTelemetry ? 'text-gray-500' : 'text-neutral-500'}>[{evt.source}]</span>
                    <span className={evt.category === 'WARN' ? (isTelemetry ? 'text-orange-400' : 'text-amber-400') : evt.category === 'HEALTHY' ? (isTelemetry ? 'text-green-400' : 'text-emerald-400') : (isTelemetry ? 'text-gray-300' : 'text-neutral-300')}>
                      {evt.message}
                    </span>
                  </div>
                ))
              ) : (
                <div className={`text-xs py-3 text-center font-mono ${isTelemetry ? 'text-gray-500' : 'text-neutral-500'}`}>No simulation events logged yet</div>
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
