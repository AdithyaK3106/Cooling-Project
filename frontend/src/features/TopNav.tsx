import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Activity, User } from 'lucide-react';

export function TopNav() {
  const navigate = useNavigate();
  const defaultMode = typeof window !== 'undefined' && window.location.pathname === '/dual-node' ? 'LOCAL' : 'LOCAL';
  const mode = localStorage.getItem('thervo_mode') || defaultMode;

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newMode = e.target.value;
    localStorage.setItem('thervo_mode', newMode);
    if (newMode === 'LOCAL') {
      navigate('/dual-node');
    } else {
      navigate('/overview');
    }
    window.location.reload();
  };

  const isLocal = mode === 'LOCAL';

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-white/[0.08] bg-[#090C12]/95 px-6 backdrop-blur-md z-40">
      {/* Brand & System Title - Click returns to Landing Page */}
      <NavLink to="/" className="flex items-center gap-2.5 hover:opacity-90 transition-opacity cursor-pointer group" title="Return to Landing Page">
        <div className="flex items-center justify-center h-8 w-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/40 text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.25)] group-hover:border-cyan-400/70 transition-all">
          <Activity size={17} className="animate-pulse" />
        </div>
        <div className="flex flex-col">
          <div className="flex items-center gap-1.5">
            <span className="font-mono font-bold tracking-wider text-white text-base">THERVO</span>
            <span className="px-1.5 py-0.2 text-[9px] font-mono font-bold uppercase tracking-widest rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">v2.4 PRO</span>
          </div>
          <span className="text-[10px] font-mono text-gray-400 tracking-tight">AI DataCenter Thermal Intelligence</span>
        </div>
      </NavLink>

      {/* Navigation Tabs */}
      <nav className="flex items-center gap-1.5 bg-black/40 p-1 rounded-lg border border-white/[0.06]">
        {isLocal ? (
          <>
            <NavItem to="/dual-node" label="Dual Node Demo" />
            <NavItem to="/pipeline" label="ML Pipeline" />
            <NavItem to="/thermal-map" label="3D Thermal Map" />
            <NavItem to="/overview" label="Datacenter Overview" />
          </>
        ) : (
          <>
            <NavItem to="/overview" label="Overview" />
            <NavItem to="/thermal-map" label="3D Thermal Map" />
            <NavItem to="/racks" label="Racks" />
            <NavItem to="/predictions" label="Predictions" />
            <NavItem to="/events" label="Events" />
            <NavItem to="/dual-node" label="Dual Node Demo" />
          </>
        )}
      </nav>

      {/* Right Controls: Mode Selector + Live Heartbeat + Status */}
      <div className="flex items-center gap-4">
        {/* Mode Selector */}
        <div className="relative">
          <select 
            value={mode}
            onChange={handleModeChange}
            className="appearance-none bg-[#131722] border border-white/10 rounded-lg px-3 py-1.5 pr-7 text-xs font-mono font-semibold text-cyan-300 outline-none cursor-pointer hover:border-cyan-500/40 hover:bg-[#181e2c] transition-all shadow-inner"
          >
            <option value="LOCAL">Local Hardware Demo</option>
            <option value="SIMULATED">Simulated Datacenter (5x5)</option>
          </select>
          <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
            <svg className="w-3 h-3 fill-current" viewBox="0 0 20 20"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/></svg>
          </div>
        </div>
        
        {/* Live Heartbeat */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-xs font-mono font-bold text-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.15)]">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500 shadow-[0_0_6px_#10b981]"></span>
          </span>
          LIVE STREAM
        </div>

        {/* System Security / User Badge */}
        <div className="flex items-center gap-2 pl-2 border-l border-white/10">
          <div className="flex items-center justify-center h-8 w-8 rounded-lg bg-white/[0.05] border border-white/10 text-gray-300 hover:text-white transition-colors cursor-pointer" title="Enterprise Console Operator">
            <User size={15} />
          </div>
        </div>
      </div>
    </header>
  );
}

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3.5 py-1.5 text-xs font-mono font-semibold rounded-md transition-all ${
          isActive 
            ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-[0_0_10px_rgba(6,182,212,0.2)]' 
            : 'text-gray-400 hover:text-gray-200 hover:bg-white/[0.04] border border-transparent'
        }`
      }
    >
      {label}
    </NavLink>
  );
}
