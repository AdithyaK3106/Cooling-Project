import { NavLink, useNavigate } from 'react-router-dom';
import { Box, User } from 'lucide-react';

export function TopNav() {
  const navigate = useNavigate();
  const mode = localStorage.getItem('thervo_mode') || 'LOCAL';

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newMode = e.target.value;
    localStorage.setItem('thervo_mode', newMode);
    if (newMode === 'LOCAL') {
      navigate('/dual-node');
    } else {
      navigate('/overview');
    }
    // Force a small reload or just let React Query refetch, but reload ensures clean state
    window.location.reload();
  };

  const isLocal = mode === 'LOCAL';

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-white/10 bg-[#0B0E14] px-6">
      <div className="flex items-center gap-2">
        <Box className="text-cyan-400" size={20} />
        <span className="font-bold tracking-wide text-white text-lg">THERVO</span>
      </div>

      <nav className="flex items-center gap-2">
        {isLocal ? (
          <NavItem to="/dual-node" label="Dual Node" />
        ) : (
          <>
            <NavItem to="/overview" label="Overview" />
            <NavItem to="/thermal-map" label="Thermal Map" />
            <NavItem to="/racks" label="Racks" />
            <NavItem to="/predictions" label="Predictions" />
            <NavItem to="/events" label="Events" />
          </>
        )}
      </nav>

      <div className="flex items-center gap-6">
        <select 
          value={mode}
          onChange={handleModeChange}
          className="bg-white/5 border border-white/10 rounded px-2 py-1 text-xs font-semibold text-cyan-400 outline-none cursor-pointer hover:bg-white/10 transition-colors"
        >
          <option value="LOCAL">Local Hardware Demo</option>
          <option value="SIMULATED">Simulated Datacenter</option>
        </select>
        
        <div className="flex items-center gap-2 text-xs font-bold text-gray-400">
          <div className="h-2 w-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e]" />
          Live
        </div>

        <div className="flex items-center justify-center h-7 w-7 rounded-full bg-white/10 text-gray-300">
          <User size={14} />
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
        `px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
          isActive ? 'bg-cyan-500/20 text-cyan-400' : 'text-gray-400 hover:bg-white/5 hover:text-white'
        }`
      }
    >
      {label}
    </NavLink>
  );
}
