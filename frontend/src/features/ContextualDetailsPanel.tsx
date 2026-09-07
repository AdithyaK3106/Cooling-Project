import { useTelemetry } from '../services/telemetryApi';
import { useUiStore } from '../stores/uiStore';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { Activity, Thermometer, Fan, AlertTriangle, CheckCircle2, Sparkles } from 'lucide-react';

export function ContextualDetailsPanel() {
  const selectedRackId = useUiStore((state) => state.selectedRackId);
  const { data: telemetry } = useTelemetry();

  if (!selectedRackId || !telemetry) {
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

  const rack = telemetry.racks?.find((r: any) => r.id === selectedRackId);
  if (!rack) return null;

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

      {/* Explainable AI (XAI) Cooling Rationale */}
      <div className="mt-4 pt-4 border-t border-white/10">
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-bold text-cyan-400 uppercase tracking-widest flex items-center gap-1.5">
            <Sparkles size={14} /> XAI Cooling Rationale
          </div>
          <span className="text-[10px] font-mono bg-cyan-500/20 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-500/30">
            {rack.ai_insights?.primary_driver || 'Multivariate Model'}
          </span>
        </div>

        {/* Natural Language AI Decision Explanation */}
        <p className="text-xs text-gray-300 bg-white/5 p-2.5 rounded-lg border border-white/5 leading-relaxed font-sans">
          {rack.ai_insights?.explanation || 'Chassis thermal risk evaluated from GNN spatial graph propagation and XGBoost tree inference.'}
        </p>

        {/* SHAP Feature Contribution Bars */}
        <div className="mt-3 space-y-2 font-mono text-[11px]">
          <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider">XAI Feature Importance Impact</div>
          {rack.ai_insights?.xai_attribution && (
            <>
              <XaiBar label="GPU Thermal Contribution" pct={rack.ai_insights.xai_attribution.gpu} color="bg-amber-500" />
              <XaiBar label="CPU Thread Contribution" pct={rack.ai_insights.xai_attribution.cpu} color="bg-blue-500" />
              <XaiBar label="GNN Spatial Heat Diffusion" pct={rack.ai_insights.xai_attribution.gnn} color="bg-purple-500" />
            </>
          )}
        </div>
      </div>

      <div className={`mt-5 rounded-lg p-3 border ${isDanger ? 'border-red-500/30 bg-red-500/10' : isWarning ? 'border-yellow-500/30 bg-yellow-500/10' : 'border-green-500/20 bg-green-500/5'}`}>
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-xs text-gray-400">Risk Score</span>
          <span className={`font-mono text-lg ${statusColor}`}>{(rack.risk_score * 100).toFixed(1)}%</span>
        </div>
        <div className={`text-xs tracking-widest uppercase ${statusColor}`}>{statusText}</div>
        
        {rack.cooling?.status === 'predictive intervention' && (
          <div className="mt-2 pt-2 border-t border-white/5 flex items-center justify-between text-cyan-400 text-xs">
            <div className="flex items-center gap-2">
              <Fan size={12} className="animate-spin" />
              <span>PREDICTIVE COOLING ACTIVE ({rack.cooling.actual_rpm} RPM)</span>
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
      <div className="flex justify-between text-gray-400">
        <span>{label}</span>
        <span className="font-bold text-white">{pct}%</span>
      </div>
      <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
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
