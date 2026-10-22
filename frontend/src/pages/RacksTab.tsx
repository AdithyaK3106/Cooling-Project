import { useState, useMemo } from 'react';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { ContextualDetailsPanel } from '../features/ContextualDetailsPanel';
import { SimulationControlBar } from '../components/simulation/SimulationControlBar';
import { Snowflake, Cpu, HardDrive, Wifi, Filter } from 'lucide-react';

export function RacksTab() {
  const { data: telemetry } = useTelemetry();
  const racks = telemetry?.racks || [];
  const { selectedRackId, setSelectedRackId } = useUiStore();

  const [filterRisk, setFilterRisk] = useState('all');
  const [filterCooling, setFilterCooling] = useState('all');
  const [sortFleet, setSortFleet] = useState('risk-desc');

  const displayList = useMemo(() => {
    let list = [...racks];

    if (filterRisk !== 'all') {
      list = list.filter((r: any) => {
        if (filterRisk === 'low') return r.risk_score < 0.35;
        if (filterRisk === 'med') return r.risk_score >= 0.35 && r.risk_score < 0.55;
        if (filterRisk === 'high') return r.risk_score >= 0.55 && r.risk_score < 0.75;
        if (filterRisk === 'crit') return r.risk_score >= 0.75;
        return true;
      });
    }

    if (filterCooling !== 'all') {
      list = list.filter((r: any) => {
        const isCool = r.cooling?.status === 'predictive intervention' || r.cooling?.override;
        return filterCooling === 'active' ? isCool : !isCool;
      });
    }

    list.sort((a: any, b: any) => {
      if (sortFleet === 'risk-desc') return b.risk_score - a.risk_score;
      if (sortFleet === 'risk-asc') return a.risk_score - b.risk_score;
      if (sortFleet === 'cpu-desc') return (b.telemetry?.cpu_util || 0) - (a.telemetry?.cpu_util || 0);
      if (sortFleet === 'gpu-desc') return (b.telemetry?.gpu_util || 0) - (a.telemetry?.gpu_util || 0);
      if (sortFleet === 'id') return a.id.localeCompare(b.id);
      return 0;
    });

    return list;
  }, [racks, filterRisk, filterCooling, sortFleet]);

  return (
    <div 
      style={{ zoom: 1.5 }}
      className="flex h-full w-full p-6 flex-col gap-6 overflow-y-auto bg-[#080A10] text-gray-200 font-sans"
    >
      <SimulationControlBar />
      
      {/* Main Fleet Layout */}
      <div className="flex h-full w-full gap-6 overflow-hidden min-h-[580px]">
        
        {/* Left: Main Fleet Table / Grid Panel */}
        <div className="flex-1 flex flex-col rounded-xl border border-white/[0.08] bg-[#0C1017]/95 backdrop-blur-md shadow-2xl overflow-hidden">
          
          {/* Panel Header & Filter Controls */}
          <div className="p-5 border-b border-white/[0.08] flex flex-wrap justify-between items-center gap-4 bg-[#0E121B]/60">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] animate-pulse" />
                <h2 className="text-base font-mono font-bold text-white tracking-wide uppercase">Rack Fleet Monitoring</h2>
              </div>
              <p className="text-xs font-mono text-gray-400 mt-0.5">
                <span className="text-cyan-300 font-bold">{displayList.length}</span> / {racks.length || 25} Racks matching operational criteria
              </p>
            </div>

            {/* Filter Dropdowns Toolbar */}
            <div className="flex items-center gap-3 flex-wrap">
              {/* Risk Filter */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] font-mono font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1">
                  <Filter size={10} className="text-cyan-400" /> Risk Filter
                </label>
                <div className="relative">
                  <select 
                    value={filterRisk} 
                    onChange={e => setFilterRisk(e.target.value)}
                    className="appearance-none bg-[#10141D] border border-white/10 rounded-lg px-3 py-1.5 pr-7 text-xs font-mono font-semibold text-gray-200 outline-none cursor-pointer hover:border-cyan-500/40 hover:bg-[#151924] transition-all shadow-inner"
                  >
                    <option value="all">All Risks</option>
                    <option value="low">Low (&lt;35%)</option>
                    <option value="med">Medium (35-55%)</option>
                    <option value="high">High (55-75%)</option>
                    <option value="crit">Critical (&gt;75%)</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                    <svg className="w-3 h-3 fill-current" viewBox="0 0 20 20"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/></svg>
                  </div>
                </div>
              </div>
              
              {/* Cooling Status Filter */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] font-mono font-bold text-gray-400 uppercase tracking-wider">Cooling Status</label>
                <div className="relative">
                  <select 
                    value={filterCooling} 
                    onChange={e => setFilterCooling(e.target.value)}
                    className="appearance-none bg-[#10141D] border border-white/10 rounded-lg px-3 py-1.5 pr-7 text-xs font-mono font-semibold text-gray-200 outline-none cursor-pointer hover:border-cyan-500/40 hover:bg-[#151924] transition-all shadow-inner"
                  >
                    <option value="all">All States</option>
                    <option value="active">Cooling Active</option>
                    <option value="inactive">Cooling Standby</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                    <svg className="w-3 h-3 fill-current" viewBox="0 0 20 20"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/></svg>
                  </div>
                </div>
              </div>

              {/* Sort By Filter */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] font-mono font-bold text-gray-400 uppercase tracking-wider">Sort By</label>
                <div className="relative">
                  <select 
                    value={sortFleet} 
                    onChange={e => setSortFleet(e.target.value)}
                    className="appearance-none bg-[#10141D] border border-white/10 rounded-lg px-3 py-1.5 pr-7 text-xs font-mono font-semibold text-gray-200 outline-none cursor-pointer hover:border-cyan-500/40 hover:bg-[#151924] transition-all shadow-inner"
                  >
                    <option value="risk-desc">Risk (High to Low)</option>
                    <option value="risk-asc">Risk (Low to High)</option>
                    <option value="cpu-desc">CPU Utilization</option>
                    <option value="gpu-desc">GPU Utilization</option>
                    <option value="id">Rack ID</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                    <svg className="w-3 h-3 fill-current" viewBox="0 0 20 20"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/></svg>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Fleet Grid */}
          <div className="flex-1 overflow-y-auto p-5">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {displayList.map((rack: any) => {
                const risk = rack.risk_score || 0;
                const riskPct = Math.round(risk * 100);
                const isSelected = rack.id === selectedRackId;
                const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;

                let cardBg = 'bg-[#10141D] border-emerald-500/30 text-emerald-300';
                let riskBadgeClass = 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
                let riskDot = 'bg-emerald-400 shadow-[0_0_6px_#10b981]';

                if (risk >= 0.75) {
                  cardBg = 'bg-red-950/25 border-red-500/50 text-red-300 shadow-[0_0_12px_rgba(239,68,68,0.15)]';
                  riskBadgeClass = 'text-red-400 bg-red-500/10 border-red-500/30';
                  riskDot = 'bg-red-400 shadow-[0_0_6px_#ef4444]';
                } else if (risk >= 0.55) {
                  cardBg = 'bg-orange-950/25 border-orange-500/50 text-orange-300 shadow-[0_0_12px_rgba(249,115,22,0.15)]';
                  riskBadgeClass = 'text-orange-400 bg-orange-500/10 border-orange-500/30';
                  riskDot = 'bg-orange-400 shadow-[0_0_6px_#f97316]';
                } else if (risk >= 0.35) {
                  cardBg = 'bg-amber-950/25 border-amber-500/50 text-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.15)]';
                  riskBadgeClass = 'text-amber-400 bg-amber-500/10 border-amber-500/30';
                  riskDot = 'bg-amber-400 shadow-[0_0_6px_#f59e0b]';
                }

                const borderClass = isSelected 
                  ? 'border-cyan-400 bg-cyan-950/30 ring-1 ring-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.3)]' 
                  : 'hover:border-white/30 hover:bg-[#141924] hover:-translate-y-0.5';

                const coolingBorder = isCooled ? 'border-l-4 border-l-cyan-400' : '';

                return (
                  <div 
                    key={rack.id}
                    onClick={() => setSelectedRackId(rack.id)}
                    className={`cursor-pointer flex flex-col p-4 rounded-xl border transition-all duration-200 min-h-[175px] justify-between font-sans ${cardBg} ${coolingBorder} ${borderClass}`}
                  >
                    {/* Rack Header */}
                    <div className="flex justify-between items-start mb-2.5 border-b border-white/[0.08] pb-2">
                      <div>
                        <h3 className="font-mono text-sm font-bold text-white flex items-center gap-1.5 tracking-wider">
                          <Cpu size={15} className="text-cyan-400" />
                          {rack.id}
                        </h3>
                        <span className="text-[10px] font-mono text-gray-400">ZONE {rack.ai_insights?.zone || 'A'} • SECTOR {rack.id}</span>
                      </div>
                      {isCooled ? (
                        <span className="bg-cyan-950/80 text-cyan-300 border border-cyan-500/50 text-[9px] font-mono font-semibold px-2 py-0.5 rounded flex items-center gap-1 animate-pulse shadow-[0_0_6px_rgba(6,182,212,0.3)]">
                          <Snowflake size={10} /> COOLING
                        </span>
                      ) : (
                        <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-white/[0.05] text-gray-400 border border-white/[0.08]">
                          Z-{rack.ai_insights?.zone || 'A'}
                        </span>
                      )}
                    </div>
                    
                    {/* Detailed Telemetry Breakdown */}
                    <div className="space-y-1.5 my-1 text-xs font-mono">
                      {/* CPU */}
                      <div>
                        <div className="flex justify-between items-center text-gray-300">
                          <span>CPU</span>
                          <span className="font-bold text-white">{rack.telemetry?.cpu_util}%</span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-[#181d28] overflow-hidden mt-0.5">
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
                        <div className="w-full h-1.5 rounded-full bg-[#181d28] overflow-hidden mt-0.5">
                          <div 
                            className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full transition-all duration-300" 
                            style={{ width: `${rack.telemetry?.gpu_util}%` }} 
                          />
                        </div>
                      </div>

                      {/* RAM & IO */}
                      <div className="flex justify-between items-center text-gray-400 text-[10px] pt-1 border-t border-white/[0.05]">
                        <span className="text-blue-300 font-semibold">RAM: {rack.telemetry?.mem_util}%</span>
                        <div className="flex items-center gap-2">
                          <span className="flex items-center gap-0.5 text-amber-300"><HardDrive size={10}/> {rack.telemetry?.disk_io}M</span>
                          <span className="flex items-center gap-0.5 text-emerald-300"><Wifi size={10}/> {rack.telemetry?.network_io}M</span>
                        </div>
                      </div>
                    </div>

                    {/* Dynamic Risk Number Footer */}
                    <div className="flex justify-between items-center mt-2 pt-2 border-t border-white/[0.08]">
                      <span className="text-[10px] font-mono text-gray-400 uppercase">THERMAL RISK</span>
                      <div className={`font-mono text-xs font-bold px-2.5 py-0.5 rounded-lg border ${riskBadgeClass} flex items-center gap-1.5 transition-all duration-300`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${riskDot}`} />
                        {riskPct}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Details Panel */}
        <div className="w-[360px] shrink-0">
          <ContextualDetailsPanel />
        </div>
      </div>

    </div>
  );
}
