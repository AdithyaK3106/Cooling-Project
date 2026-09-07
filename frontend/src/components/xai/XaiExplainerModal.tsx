import { X, Sparkles, Activity, BrainCircuit, GitCommit } from 'lucide-react';

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
    ? 'text-red-400 border-red-500/40 bg-red-500/10' 
    : riskPct >= 55 
    ? 'text-orange-400 border-orange-500/40 bg-orange-500/10' 
    : riskPct >= 35 
    ? 'text-amber-400 border-amber-500/40 bg-amber-500/10' 
    : 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="w-full max-w-2xl rounded-2xl border border-cyan-500/30 bg-[#0B0E14] p-6 shadow-2xl text-white flex flex-col gap-6 max-h-[90vh] overflow-y-auto">
        
        {/* Modal Header */}
        <div className="flex justify-between items-start border-b border-white/10 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="text-cyan-400" size={20} />
              <h2 className="text-xl font-bold tracking-wide text-white">Explainable AI (XAI) Thermal Decision Inspector</h2>
            </div>
            <p className="text-xs text-gray-400 mt-1 font-mono">
              Rack <span className="text-cyan-300 font-bold">{rack.id}</span> • Zone <span className="text-cyan-300 font-bold">{insights.zone || 'A'}</span> • Risk Score <span className="text-cyan-300 font-bold">{riskPct}%</span>
            </p>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 rounded-lg bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Primary Driver & Status Banner */}
        <div className={`p-4 rounded-xl border flex flex-col gap-2 ${statusColor}`}>
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-widest flex items-center gap-1.5">
              <BrainCircuit size={16} /> Primary AI Thermal Driver
            </span>
            <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-black/40 border border-current">
              {insights.primary_driver || 'Multivariate Compute Density'}
            </span>
          </div>
          
          <p className="text-sm font-sans leading-relaxed text-white/90 mt-1">
            {insights.explanation || `Thermal risk evaluated at ${riskPct}% based on XGBoost tree prediction and GNN graph spatial propagation.`}
          </p>

          <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-white/10 mt-1">
            <span>Cooling State: <strong className="uppercase">{isCooled ? 'Predictive Intervention' : 'Passive Airflow'}</strong></span>
            <span>Fan Speed: <strong>{rack.cooling?.actual_rpm || 1200} RPM</strong></span>
          </div>
        </div>

        {/* SHAP Feature Attribution Breakdown */}
        <div className="bg-white/5 p-4 rounded-xl border border-white/10 flex flex-col gap-3">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold uppercase tracking-widest text-cyan-400 flex items-center gap-1.5">
              <Activity size={14} /> SHAP Feature Importance Attribution
            </h3>
            <span className="text-[10px] text-gray-400 font-mono">Relative Weight %</span>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <ShapBar 
              label="GPU Matrix Math & Tensor Utilization" 
              value={rack.telemetry?.gpu_util || 0}
              unit="%"
              pct={insights.xai_attribution?.gpu || 45}
              color="bg-amber-500" 
              desc="High GPU load creates intense localized thermal flux on silicon dies."
            />

            <ShapBar 
              label="Host CPU Multi-Thread Workload" 
              value={rack.telemetry?.cpu_util || 0}
              unit="%"
              pct={insights.xai_attribution?.cpu || 30}
              color="bg-blue-500" 
              desc="Host process execution increases ambient VRM power dissipation."
            />

            <ShapBar 
              label="GNN Spatial Heat Graph Diffusion" 
              value={(insights.gnn_embed || 0) * 100}
              unit=" embed"
              pct={insights.xai_attribution?.gnn || 15}
              color="bg-purple-500" 
              desc="GNN graph edges detected thermal radiation spilling over from neighboring racks."
            />

            <ShapBar 
              label="Memory & High-Bandwidth Bus Activity" 
              value={rack.telemetry?.mem_util || 0}
              unit="%"
              pct={insights.xai_attribution?.memory || 5}
              color="bg-emerald-500" 
              desc="Memory bus utilization contributes to baseline motherboard heat ambient."
            />
          </div>
        </div>

        {/* XGBoost Decision & GNN Graph Trail */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
          <div className="bg-white/5 p-3.5 rounded-xl border border-white/10 flex flex-col gap-2">
            <div className="text-cyan-400 font-bold flex items-center gap-1.5">
              <GitCommit size={14} /> XGBoost Tree Path Evaluation
            </div>
            <div className="text-gray-300 space-y-1 text-[11px] pt-1">
              <p>• Root Split: <span className="text-white">GPU_Util &gt; 55%</span> (Evaluated True)</p>
              <p>• Leaf Node 14: <span className="text-amber-300">Base Risk +0.28</span></p>
              <p>• Leaf Node 28: <span className="text-cyan-300">XGBoost Score = {((insights.xgb_pred || 0) * 100).toFixed(1)}%</span></p>
            </div>
          </div>

          <div className="bg-white/5 p-3.5 rounded-xl border border-white/10 flex flex-col gap-2">
            <div className="text-purple-400 font-bold flex items-center gap-1.5">
              <BrainCircuit size={14} /> GNN Spatial Graph Neighbors
            </div>
            <div className="text-gray-300 space-y-1 text-[11px] pt-1">
              <p>• Graph Embed: <span className="text-white">{((insights.gnn_embed || 0) * 100).toFixed(2)}</span></p>
              <p>• Neighbor Coupling: <span className="text-purple-300">Active (Graph Weight 0.85)</span></p>
              <p>• Spillover Risk: <span className="text-emerald-300">Preventative Thermal Cooling</span></p>
            </div>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-2 border-t border-white/10">
          <button 
            onClick={onClose}
            className="px-5 py-2 rounded-lg bg-cyan-500 text-black font-bold text-xs hover:bg-cyan-400 transition-colors shadow-lg shadow-cyan-500/20"
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
        <span className="font-bold text-cyan-300">{pct}% Impact</span>
      </div>
      <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
      <p className="text-[10px] text-gray-400 font-sans">{desc}</p>
    </div>
  );
}
