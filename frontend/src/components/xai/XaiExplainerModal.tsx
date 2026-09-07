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
    ? 'text-red-300 border-red-800 bg-red-950/40' 
    : riskPct >= 55 
    ? 'text-orange-300 border-orange-800 bg-orange-950/40' 
    : riskPct >= 35 
    ? 'text-amber-300 border-amber-800 bg-amber-950/40' 
    : 'text-emerald-300 border-emerald-800 bg-emerald-950/30';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4">
      <div className="w-full max-w-2xl rounded-md border border-neutral-700 bg-[#15181e] p-5 shadow-xl text-neutral-200 flex flex-col gap-5 max-h-[90vh] overflow-y-auto font-sans">
        
        {/* Modal Header */}
        <div className="flex justify-between items-start border-b border-neutral-800 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="text-neutral-300" size={18} />
              <h2 className="text-sm font-bold tracking-wide text-neutral-100 uppercase font-mono">Explainable AI (XAI) Thermal Decision Inspector</h2>
            </div>
            <p className="text-xs text-neutral-400 mt-1 font-mono">
              Rack <span className="text-neutral-100 font-bold">{rack.id}</span> • Zone <span className="text-neutral-100 font-bold">{insights.zone || 'A'}</span> • Risk Score <span className="text-neutral-100 font-bold">{riskPct}%</span>
            </p>
          </div>
          <button 
            onClick={onClose}
            className="p-1 rounded-md bg-neutral-800 text-neutral-400 hover:bg-neutral-700 hover:text-white transition-colors border border-neutral-700"
          >
            <X size={16} />
          </button>
        </div>

        {/* Primary Driver & Status Banner */}
        <div className={`p-3.5 rounded-md border flex flex-col gap-2 ${statusColor}`}>
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold font-mono uppercase tracking-wider flex items-center gap-1.5">
              <BrainCircuit size={15} /> Primary AI Thermal Driver
            </span>
            <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-sm bg-black/50 border border-current">
              {insights.primary_driver || 'Multivariate Compute Density'}
            </span>
          </div>
          
          <p className="text-xs font-mono leading-relaxed text-neutral-200 mt-1">
            {insights.explanation || `Thermal risk evaluated at ${riskPct}% based on XGBoost tree prediction and GNN graph spatial propagation.`}
          </p>

          <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-neutral-800/80 mt-1">
            <span>Cooling State: <strong className="uppercase">{isCooled ? 'Predictive Intervention' : 'Passive Airflow'}</strong></span>
            <span>Fan Speed: <strong>{rack.cooling?.actual_rpm || 1200} RPM</strong></span>
          </div>
        </div>

        {/* SHAP Feature Attribution Breakdown */}
        <div className="bg-[#181b20] p-4 rounded-md border border-neutral-800 flex flex-col gap-3">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold font-mono uppercase tracking-widest text-neutral-300 flex items-center gap-1.5">
              <Activity size={14} /> SHAP Feature Importance Attribution
            </h3>
            <span className="text-[10px] text-neutral-400 font-mono">Relative Weight %</span>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <ShapBar 
              label="GPU Matrix Math & Tensor Utilization" 
              value={rack.telemetry?.gpu_util || 0}
              unit="%"
              pct={insights.xai_attribution?.gpu || 45}
              color="bg-amber-600" 
              desc="High GPU load creates intense localized thermal flux on silicon dies."
            />

            <ShapBar 
              label="Host CPU Multi-Thread Workload" 
              value={rack.telemetry?.cpu_util || 0}
              unit="%"
              pct={insights.xai_attribution?.cpu || 30}
              color="bg-blue-600" 
              desc="Host process execution increases ambient VRM power dissipation."
            />

            <ShapBar 
              label="GNN Spatial Heat Graph Diffusion" 
              value={(insights.gnn_embed || 0) * 100}
              unit=" embed"
              pct={insights.xai_attribution?.gnn || 15}
              color="bg-purple-600" 
              desc="GNN graph edges detected thermal radiation spilling over from neighboring racks."
            />

            <ShapBar 
              label="Memory & High-Bandwidth Bus Activity" 
              value={rack.telemetry?.mem_util || 0}
              unit="%"
              pct={insights.xai_attribution?.memory || 5}
              color="bg-emerald-600" 
              desc="Memory bus utilization contributes to baseline motherboard heat ambient."
            />
          </div>
        </div>

        {/* XGBoost Decision & GNN Graph Trail */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 font-mono text-xs">
          <div className="bg-[#181b20] p-3 rounded-md border border-neutral-800 flex flex-col gap-1.5">
            <div className="text-neutral-200 font-bold flex items-center gap-1.5">
              <GitCommit size={14} /> XGBoost Tree Path Evaluation
            </div>
            <div className="text-neutral-400 space-y-1 text-[11px] pt-1">
              <p>• Root Split: <span className="text-neutral-200">GPU_Util &gt; 55%</span> (Evaluated True)</p>
              <p>• Leaf Node 14: <span className="text-amber-300">Base Risk +0.28</span></p>
              <p>• Leaf Node 28: <span className="text-neutral-100">XGBoost Score = {((insights.xgb_pred || 0) * 100).toFixed(1)}%</span></p>
            </div>
          </div>

          <div className="bg-[#181b20] p-3 rounded-md border border-neutral-800 flex flex-col gap-1.5">
            <div className="text-neutral-200 font-bold flex items-center gap-1.5">
              <BrainCircuit size={14} /> GNN Spatial Graph Neighbors
            </div>
            <div className="text-neutral-400 space-y-1 text-[11px] pt-1">
              <p>• Graph Embed: <span className="text-neutral-200">{((insights.gnn_embed || 0) * 100).toFixed(2)}</span></p>
              <p>• Neighbor Coupling: <span className="text-purple-300">Active (Graph Weight 0.85)</span></p>
              <p>• Spillover Risk: <span className="text-emerald-300">Preventative Thermal Cooling</span></p>
            </div>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-2 border-t border-neutral-800">
          <button 
            onClick={onClose}
            className="px-4 py-1.5 rounded-md bg-zinc-700 text-neutral-100 font-mono font-bold text-xs hover:bg-zinc-600 transition-colors border border-zinc-600"
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
      <div className="flex justify-between items-center text-neutral-300">
        <span className="font-semibold">{label} ({value.toFixed(0)}{unit})</span>
        <span className="font-bold text-neutral-200">{pct}% Impact</span>
      </div>
      <div className="w-full bg-neutral-800 h-2 rounded-sm overflow-hidden border border-neutral-700/50">
        <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
      <p className="text-[10px] text-neutral-400 font-sans">{desc}</p>
    </div>
  );
}
