import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { getSimulatedTelemetry } from '../services/simulation';
import { Activity, Thermometer, Fan, AlertTriangle, CheckCircle2, ChevronRight } from 'lucide-react';

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

  if (!selectedRackId || !rack) {
    return (
      <div className="w-80 rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur-xl shadow-2xl transition-all duration-300">
        <h2 className="mb-3 text-sm font-bold tracking-widest text-gray-500 uppercase">System Status</h2>
        <div className="flex items-center gap-3 text-[#4ade80]">
          <CheckCircle2 size={24} />
          <span className="text-lg font-mono">ALL SYSTEMS NOMINAL</span>
        </div>
        <p className="mt-4 text-xs text-gray-400">Select a rack in the 3D twin to view real-time diagnostics and thermal state.</p>
      </div>
    );
  }

  const isDanger = rack.risk_score > 0.7;
  const isWarning = rack.risk_score > 0.4 && !isDanger;
  const statusColor = isDanger ? 'text-[#f87171]' : isWarning ? 'text-[#fbbf24]' : 'text-[#4ade80]';
  const statusText = isDanger ? 'CRITICAL RISK' : isWarning ? 'ELEVATED RISK' : 'LOW RISK';

  return (
    <div className="w-80 rounded-xl border border-white/10 bg-[#0B0E14]/80 p-5 backdrop-blur-xl shadow-2xl transition-all duration-300">
      <div className="flex items-center justify-between mb-4 border-b border-white/10 pb-3">
        <div>
          <h2 className="text-xl font-mono text-white tracking-wider">RACK {rack.id}</h2>
          <div className="flex items-center gap-1.5 mt-1">
            <div className={`w-2 h-2 rounded-full ${isDanger ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
            <span className="text-xs text-gray-400 uppercase tracking-widest">ONLINE</span>
          </div>
        </div>
        <div className={`flex items-center justify-center w-10 h-10 rounded-full border border-white/5 bg-white/5 ${statusColor}`}>
          <AlertTriangle size={18} />
        </div>
      </div>
      
      <div className="space-y-4 font-mono text-sm">
        <MetricRow icon={<Activity size={16}/>} label="CPU" value={rack.telemetry.cpu_util} unit="%" />
        <MetricRow icon={<Activity size={16}/>} label="GPU" value={rack.telemetry.gpu_util} unit="%" />
        <MetricRow icon={<Thermometer size={16}/>} label="Temperature" value={rack.telemetry.cpu_temp} unit="°C" highlight={isDanger || isWarning} />
      </div>

      <div className="mt-4 pt-4 border-t border-white/10 space-y-4 font-mono text-sm">
        <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">AI Insights</div>
        <MetricRow icon={<Activity size={16}/>} label="XGBoost Pred" value={rack.ai_insights?.xgb_pred ? rack.ai_insights.xgb_pred * 100 : 0} unit="%" />
        <MetricRow icon={<Activity size={16}/>} label="GNN Embed" value={rack.ai_insights?.gnn_embed ? rack.ai_insights.gnn_embed * 100 : 0} unit="%" format={(v: number) => v.toFixed(2)} />
      </div>

      <div className={`mt-6 rounded-lg p-3 border ${isDanger ? 'border-red-500/30 bg-red-500/10' : isWarning ? 'border-yellow-500/30 bg-yellow-500/10' : 'border-green-500/20 bg-green-500/5'}`}>
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-xs text-gray-400">Risk Score</span>
          <span className={`font-mono text-lg ${statusColor}`}>{(rack.risk_score * 100).toFixed(1)}%</span>
        </div>
        <div className={`text-xs tracking-widest uppercase ${statusColor}`}>{statusText}</div>
        
        {rack.cooling?.status === 'predictive intervention' && (
          <div className="mt-2 pt-2 border-t border-white/5 flex items-center justify-between text-cyan-400 text-xs">
            <div className="flex items-center gap-2">
              <Fan size={12} className="animate-spin" />
              <span>PREDICTIVE COOLING ACTIVE</span>
            </div>
          </div>
        )}
      </div>

      <button className="mt-4 w-full flex items-center justify-between rounded bg-white/5 px-4 py-2 text-xs font-bold text-gray-300 transition-colors hover:bg-white/10 hover:text-white group">
        <span>VIEW DETAILS</span>
        <ChevronRight size={14} className="transition-transform group-hover:translate-x-1" />
      </button>
    </div>
  );
}

function MetricRow({ icon, label, value, unit, highlight = false, format }: any) {
  const displayValue = useAnimatedNumber(value, 400, format);
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-gray-400">
        <span className="opacity-70">{icon}</span>
        <span>{label}</span>
      </div>
      <span className={`font-bold ${highlight ? 'text-orange-400' : 'text-cyan-400'}`}>
        {displayValue}{unit}
      </span>
    </div>
  );
}
