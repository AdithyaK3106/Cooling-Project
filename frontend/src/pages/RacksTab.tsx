import { useState, useMemo } from 'react';
import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { ContextualDetailsPanel } from '../features/ContextualDetailsPanel';

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
      list = list.filter(r => {
        if (filterRisk === 'low') return r.risk_score < 0.35;
        if (filterRisk === 'med') return r.risk_score >= 0.35 && r.risk_score < 0.55;
        if (filterRisk === 'high') return r.risk_score >= 0.55 && r.risk_score < 0.75;
        if (filterRisk === 'crit') return r.risk_score >= 0.75;
        return true;
      });
    }

    if (filterCooling !== 'all') {
      list = list.filter(r => {
        const isCool = r.cooling?.status === 'predictive intervention';
        return filterCooling === 'active' ? isCool : !isCool;
      });
    }

    list.sort((a, b) => {
      if (sortFleet === 'risk-desc') return b.risk_score - a.risk_score;
      if (sortFleet === 'risk-asc') return a.risk_score - b.risk_score;
      if (sortFleet === 'cpu-desc') return b.telemetry.cpu_util - a.telemetry.cpu_util;
      if (sortFleet === 'gpu-desc') return b.telemetry.gpu_util - a.telemetry.gpu_util;
      if (sortFleet === 'id') return a.id.localeCompare(b.id);
      return 0;
    });

    return list;
  }, [racks, filterRisk, filterCooling, sortFleet]);

  return (
    <div className="flex h-full w-full p-6 gap-6 overflow-hidden">
      
      {/* Main Fleet Column */}
      <div className="flex-1 flex flex-col rounded-xl border border-white/10 bg-[#0B0E14]/80 backdrop-blur shadow-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10 flex flex-wrap justify-between items-center gap-4">
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide">Rack Fleet</h2>
            <p className="text-sm text-gray-400">{displayList.length} racks matching criteria</p>
          </div>

          <div className="flex gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Risk</label>
              <select 
                value={filterRisk} 
                onChange={e => setFilterRisk(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm text-white outline-none"
              >
                <option value="all">All</option>
                <option value="low">Low (&lt;35%)</option>
                <option value="med">Medium (35-55%)</option>
                <option value="high">High (55-75%)</option>
                <option value="crit">Critical (&gt;75%)</option>
              </select>
            </div>
            
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Cooling</label>
              <select 
                value={filterCooling} 
                onChange={e => setFilterCooling(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm text-white outline-none"
              >
                <option value="all">All</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Sort By</label>
              <select 
                value={sortFleet} 
                onChange={e => setSortFleet(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm text-white outline-none"
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
            {displayList.map(rack => {
              const isSelected = rack.id === selectedRackId;
              const isDanger = rack.risk_score > 0.7;
              const isWarning = rack.risk_score > 0.4 && !isDanger;
              const borderClass = isSelected ? 'border-cyan-400 ring-1 ring-cyan-400' : 'border-white/10 hover:border-white/30';
              const riskColor = isDanger ? 'text-red-400 bg-red-400/10' : isWarning ? 'text-yellow-400 bg-yellow-400/10' : 'text-green-400 bg-green-400/10';

              return (
                <div 
                  key={rack.id}
                  onClick={() => setSelectedRackId(rack.id)}
                  className={`cursor-pointer flex flex-col p-4 rounded-lg bg-white/5 transition-all ${borderClass}`}
                >
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="font-mono text-lg font-bold text-white">{rack.id}</h3>
                      <span className="text-xs text-gray-400">Zone {rack.ai_insights?.zone || '?'}</span>
                    </div>
                    {rack.cooling?.status === 'predictive intervention' && (
                      <span className="bg-cyan-500/20 text-cyan-400 text-[10px] font-bold px-2 py-1 rounded">❄ ACTIVE</span>
                    )}
                  </div>
                  
                  <div className="flex justify-between items-end mt-auto">
                    <div className="flex flex-col gap-1 text-xs font-mono text-gray-400">
                      <span>CPU: {rack.telemetry.cpu_util.toFixed(0)}%</span>
                      <span>GPU: {rack.telemetry.gpu_util.toFixed(0)}%</span>
                    </div>
                    <div className={`font-mono text-xl font-bold px-2 py-1 rounded ${riskColor}`}>
                      {(rack.risk_score * 100).toFixed(0)}%
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
  );
}
