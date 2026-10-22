import { X, Sparkles, Activity, BrainCircuit, GitCommit, Wind } from 'lucide-react';

export interface XaiModalProps {
  rack: any;
  isOpen: boolean;
  onClose: () => void;
}

export function XaiExplainerModal({ rack, isOpen, onClose }: XaiModalProps) {
  if (!isOpen || !rack) return null;

  const insights = rack.ai_insights || {};
  const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;
  const riskPct = Math.round((rack.risk_score || 0) * 100);

  const statusColor = riskPct >= 75 
    ? 'text-red-300 border-red-500/40 bg-red-950/30' 
    : riskPct >= 55 
    ? 'text-orange-300 border-orange-500/40 bg-orange-950/30' 
    : riskPct >= 35 
    ? 'text-amber-300 border-amber-500/40 bg-amber-950/30' 
    : 'text-emerald-300 border-emerald-500/40 bg-emerald-950/25';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="w-full max-w-2xl rounded-2xl border border-white/[0.12] bg-[#0C1017] p-6 shadow-[0_0_50px_rgba(0,0,0,0.8)] text-gray-200 flex flex-col gap-5 max-h-[90vh] overflow-y-auto font-sans">
        
        {/* Modal Header */}
        <div className="flex justify-between items-start border-b border-white/[0.08] pb-4">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_10px_rgba(6,182,212,0.2)]">
                <Sparkles size={16} />
              </div>
              <h2 className="text-sm font-mono font-bold tracking-wide text-white uppercase">Explainable AI (XAI) Decision Inspector</h2>
            </div>
            <p className="text-xs text-gray-400 mt-1 font-mono">
              Rack <span className="text-cyan-400 font-bold">{rack.id}</span> • Sector <span className="text-white font-bold">{insights.zone || 'A'}</span> • AI Risk Score <span className="text-white font-bold">{riskPct}%</span>
            </p>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 rounded-lg bg-white/[0.05] text-gray-400 hover:bg-white/[0.1] hover:text-white transition-all border border-white/10"
          >
            <X size={16} />
          </button>
        </div>

        {/* Primary Driver & Status Banner */}
        <div className={`p-4 rounded-xl border flex flex-col gap-2.5 ${statusColor}`}>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <span className="text-xs font-bold font-mono uppercase tracking-wider flex items-center gap-1.5">
              <BrainCircuit size={15} /> Primary AI Thermal Driver
            </span>
            <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded bg-black/60 border border-current">
              {insights.primary_driver || 'Multivariate Compute Density'}
            </span>
          </div>
          
          <p className="text-xs font-mono leading-relaxed text-gray-200">
            {insights.explanation || `Thermal risk evaluated at ${riskPct}% based on XGBoost gradient tree prediction and GNN spatial graph propagation.`}
          </p>

          <div className="flex items-center justify-between text-xs font-mono pt-2.5 border-t border-white/[0.08] mt-0.5 flex-wrap gap-2">
            <span className="flex items-center gap-1.5">
              <Wind size={13} className="text-cyan-400" />
              Cooling State: <strong className="uppercase text-white">{isCooled ? 'Predictive Intervention' : 'Passive Airflow'}</strong>
            </span>
            <span>Fan Speed: <strong className="text-cyan-300">{rack.cooling?.actual_rpm || 1200} RPM</strong></span>
          </div>
        </div>

        {/* SHAP Feature Attribution Breakdown */}
        <div className="bg-[#10141D] p-4.5 rounded-xl border border-white/[0.08] flex flex-col gap-3.5">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold font-mono uppercase tracking-widest text-gray-300 flex items-center gap-2">
              <Activity size={14} className="text-cyan-400" /> SHAP Feature Importance Attribution
            </h3>
            <span className="text-[10px] text-gray-400 font-mono">Relative Weight %</span>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <ShapBar 
              label="GPU Matrix Math & Tensor Utilization" 
              value={rack.telemetry?.gpu_util || 0}
              unit="%"
              pct={insights.xai_attribution?.gpu || 45}
              color="bg-gradient-to-r from-amber-500 to-orange-500" 
              desc="High GPU compute load creates intense localized thermal flux on silicon dies."
            />

            <ShapBar 
              label="Host CPU Multi-Thread Workload" 
              value={rack.telemetry?.cpu_util || 0}
              unit="%"
              pct={insights.xai_attribution?.cpu || 30}
              color="bg-gradient-to-r from-cyan-500 to-blue-500" 
              desc="Host process execution increases ambient VRM power dissipation."
            />

            <ShapBar 
              label="GNN Spatial Heat Graph Diffusion" 
              value={(insights.gnn_embed || 0) * 100}
              unit=" embed"
              pct={insights.xai_attribution?.gnn || 15}
              color="bg-gradient-to-r from-purple-500 to-indigo-500" 
              desc="GNN graph edges detected thermal radiation spilling over from neighboring racks."
            />

            <ShapBar 
              label="Memory & High-Bandwidth Bus Activity" 
              value={rack.telemetry?.mem_util || 0}
              unit="%"
              pct={insights.xai_attribution?.memory || 5}
              color="bg-gradient-to-r from-emerald-500 to-teal-500" 
              desc="Memory bus utilization contributes to baseline motherboard heat ambient."
            />
          </div>
        </div>

        {/* XGBoost Decision & GNN Graph Trail */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 font-mono text-xs">
          <div className="bg-[#10141D] p-3.5 rounded-xl border border-white/[0.08] flex flex-col gap-1.5">
            <div className="text-white font-bold flex items-center gap-1.5">
              <GitCommit size={14} className="text-cyan-400" /> XGBoost Tree Path Evaluation
            </div>
            <div className="text-gray-400 space-y-1 text-[11px] pt-1">
              <p>• Root Split: <span className="text-cyan-300">GPU_Util &gt; 55%</span> (True)</p>
              <p>• Leaf Node 14: <span className="text-amber-300">Base Risk +0.28</span></p>
              <p>• Final Score: <span className="text-white font-bold">{((insights.xgb_pred || 0) * 100).toFixed(1)}%</span></p>
            </div>
          </div>

          <div className="bg-[#10141D] p-3.5 rounded-xl border border-white/[0.08] flex flex-col gap-1.5">
            <div className="text-white font-bold flex items-center gap-1.5">
              <BrainCircuit size={14} className="text-purple-400" /> GNN Spatial Graph Neighbors
            </div>
            <div className="text-gray-400 space-y-1 text-[11px] pt-1">
              <p>• Graph Embed: <span className="text-white">{((insights.gnn_embed || 0) * 100).toFixed(2)}</span></p>
              <p>• Neighbor Coupling: <span className="text-purple-300">Active (Weight 0.85)</span></p>
              <p>• Spillover Risk: <span className="text-emerald-300">Predictive Cooling</span></p>
            </div>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-3 border-t border-white/[0.08]">
          <button 
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-white/[0.08] text-white font-mono font-bold text-xs hover:bg-white/[0.15] transition-all border border-white/10"
          >
            Close Inspector
          </button>
        </div>

      </div>
    </div>
  );
}

function ShapBar({ label, value, unit, pct, color, desc }: { label: string; value: number; unit: string; pct: number; color: string; desc: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center text-gray-300">
        <span className="font-semibold">{label} ({value.toFixed(0)}{unit})</span>
        <span className="font-bold text-white bg-white/[0.05] px-1.5 py-0.2 rounded border border-white/[0.08]">{pct}% Impact</span>
      </div>
      <div className="w-full bg-[#181d28] h-2 rounded-full overflow-hidden border border-white/[0.05]">
        <div className={`h-full ${color} rounded-full transition-all duration-300`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
      <p className="text-[10px] text-gray-400 font-sans">{desc}</p>
    </div>
  );
}
