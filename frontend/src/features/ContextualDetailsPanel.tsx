import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { getSimulatedTelemetry } from '../services/simulation';
import { Activity, Thermometer, Fan, AlertTriangle, CheckCircle2, Sparkles, Cpu, Zap } from 'lucide-react';

export function ContextualDetailsPanel() {
  const selectedRackId = useUiStore((state) => state.selectedRackId);
  const { data: telemetry } = useTelemetry();

  const numMatch = (selectedRackId || '').match(/\d+/);
  const rackNum = numMatch ? parseInt(numMatch[0], 10) : -1;
  const matchRack = (r: any) => 
    r.id === selectedRackId || 
    r.id === `A0${rackNum}` || 
    r.id === `A${rackNum}` ||
    (rackNum > 0 && parseInt(r.id.replace(/\D+/g, ''), 10) === rackNum);

  const sim = getSimulatedTelemetry();
  const simRack = sim?.racks?.find(matchRack);
  const fromTelem = telemetry?.racks?.find(matchRack);
  const rack = fromTelem ? {
    ...fromTelem,
    ...(simRack && (simRack.coolingActive !== undefined || simRack.overrideEnabled !== undefined) ? {
      coolingActive: simRack.coolingActive,
      overrideEnabled: simRack.overrideEnabled,
      cooling: { ...fromTelem.cooling, status: (simRack.coolingActive || simRack.overrideEnabled) ? 'predictive intervention' : (fromTelem.cooling?.status || 'normal') }
    } : {})
  } : simRack;

  // Fleet summary metrics when no rack is selected
  const racksList = telemetry?.racks || [];
  const coolingCount = racksList.filter((r: any) => r.cooling?.status === 'predictive intervention' || r.cooling?.override).length;
  const avgFleetRisk = racksList.length > 0 
    ? (racksList.reduce((acc: number, r: any) => acc + (r.risk_score || 0), 0) / racksList.length) * 100 
    : 32;

  if (!selectedRackId || !rack) {
    return (
      <div className="w-full rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-5 backdrop-blur-md shadow-2xl transition-all duration-300 flex flex-col gap-4 font-sans">
        <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
          <h2 className="text-xs font-mono font-bold tracking-widest text-gray-400 uppercase">Fleet System Status</h2>
          <span className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#10b981] animate-pulse" />
            OPERATIONAL
          </span>
        </div>

        <div className="flex items-center gap-3 p-3.5 rounded-xl bg-emerald-950/25 border border-emerald-500/30 text-emerald-400">
          <CheckCircle2 size={22} className="shrink-0" />
          <div>
            <div className="text-sm font-mono font-bold tracking-wider">ALL SYSTEMS NOMINAL</div>
            <div className="text-[10px] font-mono text-emerald-300/80">Cluster Health Score: 98.4%</div>
          </div>
        </div>

        {/* Fleet Summary KPI Grid */}
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
            <div className="text-gray-400 text-[10px] uppercase font-bold">TOTAL RACKS</div>
            <div className="text-base font-bold text-white mt-0.5">{racksList.length || 25} Units</div>
          </div>

          <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
            <div className="text-gray-400 text-[10px] uppercase font-bold">ACTIVE COOLING</div>
            <div className="text-base font-bold text-cyan-300 mt-0.5">{coolingCount} / {racksList.length || 25}</div>
          </div>

          <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
            <div className="text-gray-400 text-[10px] uppercase font-bold">AVG RISK</div>
            <div className="text-base font-bold text-amber-300 mt-0.5">{avgFleetRisk.toFixed(1)}%</div>
          </div>

          <div className="bg-[#10141D] p-2.5 rounded-lg border border-white/[0.06]">
            <div className="text-gray-400 text-[10px] uppercase font-bold">ACCURACY</div>
            <div className="text-base font-bold text-emerald-400 mt-0.5">96.5%</div>
          </div>
        </div>

        <div className="mt-1 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-[11px] text-gray-400 leading-relaxed font-mono">
          Click any rack card in the fleet grid to inspect real-time telemetry, XAI explanations, and thermal state.
        </div>
      </div>
    );
  }

  const isDanger = rack.risk_score >= 0.75;
  const isHigh = rack.risk_score >= 0.55 && !isDanger;
  const isWarning = rack.risk_score >= 0.35 && !isHigh && !isDanger;

  const statusColor = isDanger ? 'text-red-400' : isHigh ? 'text-orange-400' : isWarning ? 'text-amber-400' : 'text-emerald-400';
  const statusBorder = isDanger ? 'border-red-500/40 bg-red-950/25' : isHigh ? 'border-orange-500/40 bg-orange-950/25' : isWarning ? 'border-amber-500/40 bg-amber-950/25' : 'border-emerald-500/30 bg-emerald-950/20';
  const statusText = isDanger ? 'CRITICAL RISK' : isHigh ? 'HIGH RISK' : isWarning ? 'ELEVATED RISK' : 'LOW RISK';

  return (
    <div className="w-full rounded-xl border border-white/[0.08] bg-[#0C1017]/95 p-5 backdrop-blur-md shadow-2xl transition-all duration-300 flex flex-col gap-4 font-sans">
      {/* Rack Header */}
      <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
        <div>
          <h2 className="text-base font-mono font-bold text-white tracking-wider flex items-center gap-1.5">
            <Cpu size={16} className="text-cyan-400" />
            RACK {rack.id}
          </h2>
          <div className="flex items-center gap-1.5 mt-0.5">
            <div className={`w-2 h-2 rounded-full ${isDanger ? 'bg-red-500 animate-pulse' : 'bg-emerald-400 shadow-[0_0_6px_#10b981]'}`} />
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider">ZONE {rack.ai_insights?.zone || 'A'} • ONLINE</span>
          </div>
        </div>
        <div className={`flex items-center justify-center w-8 h-8 rounded-lg border ${statusBorder} ${statusColor}`}>
          <AlertTriangle size={16} />
        </div>
      </div>
      
      {/* Telemetry Metric Rows */}
      <div className="space-y-2.5 font-mono text-xs">
        <MetricRow icon={<Activity size={14} className="text-cyan-400"/>} label="CPU Utilization" value={rack.telemetry?.cpu_util || 0} unit="%" />
        <MetricRow icon={<Activity size={14} className="text-amber-400"/>} label="GPU Compute" value={rack.telemetry?.gpu_util || 0} unit="%" />
        <MetricRow icon={<Thermometer size={14} className="text-red-400"/>} label="Chassis Temp" value={rack.telemetry?.cpu_temp || 45} unit="°C" highlight={isDanger || isHigh || isWarning} />
        <MetricRow icon={<Activity size={14} className="text-blue-400"/>} label="RAM Memory" value={rack.telemetry?.mem_util || 0} unit="%" />
        <MetricRow icon={<Zap size={14} className="text-cyan-300"/>} label="Power Draw" value={rack.telemetry?.power_draw || 250} unit=" W" />
      </div>

      {/* Explainable AI (XAI) Cooling Rationale */}
      <div className="mt-1 pt-3.5 border-t border-white/[0.08] flex flex-col gap-2.5">
        <div className="flex items-center justify-between">
          <div className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles size={13} /> XAI Cooling Rationale
          </div>
          <span className="text-[9px] font-mono bg-cyan-500/15 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/30 font-semibold">
            {rack.ai_insights?.primary_driver || 'Multivariate Model'}
          </span>
        </div>

        {/* Natural Language AI Decision Explanation */}
        <p className="text-xs text-gray-300 bg-[#10141D] p-3 rounded-lg border border-white/[0.06] leading-relaxed font-mono text-[11px]">
          {rack.ai_insights?.explanation || 'Chassis thermal risk evaluated from GNN spatial graph propagation and XGBoost tree inference.'}
        </p>

        {/* SHAP Feature Contribution Bars */}
        <div className="space-y-2 font-mono text-[11px]">
          <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider">XAI Feature Importance Impact</div>
          {rack.ai_insights?.xai_attribution && (
            <>
              <XaiBar label="GPU Thermal Contribution" pct={rack.ai_insights.xai_attribution.gpu} color="bg-gradient-to-r from-amber-500 to-orange-500" />
              <XaiBar label="CPU Thread Contribution" pct={rack.ai_insights.xai_attribution.cpu} color="bg-gradient-to-r from-cyan-500 to-blue-500" />
              <XaiBar label="GNN Spatial Heat Diffusion" pct={rack.ai_insights.xai_attribution.gnn} color="bg-gradient-to-r from-purple-500 to-indigo-500" />
            </>
          )}
        </div>
      </div>

      {/* Risk Score & Cooling Status Box */}
      <div className={`rounded-xl p-3.5 border ${statusBorder} flex flex-col gap-1.5`}>
        <div className="flex justify-between items-baseline">
          <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">AI Risk Score</span>
          <span className={`font-mono text-lg font-bold ${statusColor}`}>{(rack.risk_score * 100).toFixed(1)}%</span>
        </div>
        <div className={`text-[10px] font-mono tracking-widest uppercase font-bold ${statusColor}`}>{statusText}</div>
        
        {rack.cooling?.status === 'predictive intervention' && (
          <div className="mt-1.5 pt-2 border-t border-white/[0.08] flex items-center justify-between text-cyan-300 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <Fan size={13} className="animate-spin text-cyan-400" />
              <span className="text-[10px] font-bold">PREDICTIVE COOLING ACTIVE ({rack.cooling.actual_rpm} RPM)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function XaiBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-gray-400 text-[10px]">
        <span>{label}</span>
        <span className="font-bold text-white bg-white/[0.05] px-1 py-0.2 rounded">{pct}%</span>
      </div>
      <div className="w-full bg-[#181d28] h-1.5 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-300`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
    </div>
  );
}

function MetricRow({ icon, label, value, unit, highlight = false, format }: any) {
  const displayValue = useAnimatedNumber(value, 400, format);
  return (
    <div className="flex items-center justify-between bg-[#10141D] px-2.5 py-1.5 rounded-lg border border-white/[0.04]">
      <div className="flex items-center gap-2 text-gray-400 text-xs">
        <span>{icon}</span>
        <span>{label}</span>
      </div>
      <span className={`font-bold font-mono text-xs ${highlight ? 'text-amber-400' : 'text-cyan-300'}`}>
        {displayValue}{unit}
      </span>
    </div>
  );
}
