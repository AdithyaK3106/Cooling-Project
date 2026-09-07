import { useState, useMemo } from 'react';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { ContextualDetailsPanel } from '../features/ContextualDetailsPanel';
import { SimulationControlBar } from '../components/simulation/SimulationControlBar';
import { Snowflake, Cpu, HardDrive, Wifi } from 'lucide-react';

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
    <div className="flex h-full w-full p-6 flex-col gap-6 overflow-y-auto bg-[#07090E] text-white">
      <SimulationControlBar />
      
      {/* Main Fleet Layout */}
      <div className="flex h-full w-full gap-6 overflow-hidden">
      <div className="flex-1 flex flex-col rounded-xl border border-white/10 bg-[#0B0E14]/80 backdrop-blur shadow-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10 flex flex-wrap justify-between items-center gap-4">
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide">Rack Fleet Monitoring</h2>
            <p className="text-sm text-gray-400">{displayList.length} / 25 racks matching operational criteria</p>
          </div>

          <div className="flex gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Risk Filter</label>
              <select 
                value={filterRisk} 
                onChange={e => setFilterRisk(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm text-white outline-none cursor-pointer"
              >
                <option value="all">All Risks</option>
                <option value="low">Low (&lt;35%)</option>
                <option value="med">Medium (35-55%)</option>
                <option value="high">High (55-75%)</option>
                <option value="crit">Critical (&gt;75%)</option>
              </select>
            </div>
            
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Cooling Status</label>
              <select 
                value={filterCooling} 
                onChange={e => setFilterCooling(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm text-white outline-none cursor-pointer"
              >
                <option value="all">All States</option>
                <option value="active">Cooling Active</option>
                <option value="inactive">Cooling Standby</option>
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Sort By</label>
              <select 
                value={sortFleet} 
                onChange={e => setSortFleet(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm text-white outline-none cursor-pointer"
              >
                <option value="risk-desc">Risk (High to Low)</option>
                <option value="risk-asc">Risk (Low to High)</option>
                <option value="cpu-desc">CPU Utilization</option>
                <option value="gpu-desc">GPU Utilization</option>
                <option value="id">Rack ID</option>
              </select>
            </div>
          </div>
        </div>

        {/* Fleet Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {displayList.map((rack: any) => {
              const risk = rack.risk_score || 0;
              const riskPct = Math.round(risk * 100);
              const isSelected = rack.id === selectedRackId;
              const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;

              let cardBg = 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300';
              let riskBadgeClass = 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30';

              if (risk >= 0.75) {
                cardBg = 'bg-red-950/50 border-red-500/70 text-red-200 animate-pulse';
                riskBadgeClass = 'text-red-400 bg-red-400/10 border-red-500/30';
              } else if (risk >= 0.55) {
                cardBg = 'bg-orange-950/40 border-orange-500/60 text-orange-200';
                riskBadgeClass = 'text-orange-400 bg-orange-400/10 border-orange-500/30';
              } else if (risk >= 0.35) {
                cardBg = 'bg-amber-950/35 border-amber-500/50 text-amber-200';
                riskBadgeClass = 'text-amber-400 bg-amber-400/10 border-amber-500/30';
              }

              const borderClass = isSelected ? 'border-cyan-400 ring-2 ring-cyan-400/80 shadow-lg shadow-cyan-500/20' : 'hover:border-cyan-400/50';

              return (
                <div 
                  key={rack.id}
                  onClick={() => setSelectedRackId(rack.id)}
                  className={`cursor-pointer flex flex-col p-4 rounded-xl border backdrop-blur transition-all duration-300 ${cardBg} ${borderClass}`}
                >
                  <div className="flex justify-between items-start mb-3 border-b border-white/10 pb-2">
                    <div>
                      <h3 className="font-mono text-base font-bold text-white flex items-center gap-1.5">
                        <Cpu size={16} className="text-cyan-400" />
                        {rack.id}
                      </h3>
                      <span className="text-[10px] font-mono text-gray-400">ZONE {rack.ai_insights?.zone || '?'}</span>
                    </div>
                    {isCooled && (
                      <span className="bg-cyan-400/20 text-cyan-300 border border-cyan-400/40 text-[9px] font-bold px-2 py-0.5 rounded flex items-center gap-1">
                        <Snowflake size={10} className="animate-spin" /> COOLING
                      </span>
                    )}
                  </div>
                  
                  {/* Detailed Telemetry Breakdown */}
                  <div className="space-y-1.5 my-2 text-xs font-mono">
                    <div className="flex justify-between items-center text-gray-300">
                      <span>CPU</span>
                      <span className="font-bold text-white">{rack.telemetry?.cpu_util}%</span>
                    </div>
                    <div className="w-full h-1 bg-black/40 rounded-full overflow-hidden">
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

                    <div className="flex justify-between items-center text-gray-400 text-[10px] pt-1">
                      <span className="flex items-center gap-1"><HardDrive size={10}/> D: {rack.telemetry?.disk_io}</span>
                      <span className="flex items-center gap-1"><Wifi size={10}/> N: {rack.telemetry?.network_io}</span>
                    </div>
                  </div>

                  {/* Dynamic Risk Number (Changes with tick updates!) */}
                  <div className="flex justify-between items-end mt-2 pt-2 border-t border-white/10">
                    <span className="text-[10px] font-mono text-gray-400">THERMAL RISK</span>
                    <div className={`font-mono text-lg font-bold px-2.5 py-0.5 rounded-lg border ${riskBadgeClass} transition-all duration-300`}>
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
      <div className="w-80 shrink-0">
        <ContextualDetailsPanel />
      </div>
      </div>

    </div>
  );
}
