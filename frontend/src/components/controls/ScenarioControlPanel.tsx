import { useState } from 'react';
import { useUpdateSimulationControls } from '../../services/controlApi';
import { Settings2 } from 'lucide-react';

export function ScenarioControlPanel() {
  const [load, setLoad] = useState(50);
  const [offset, setOffset] = useState(0);
  const [mode, setMode] = useState<'LOCAL_LAPTOP' | 'DATA_CENTER_SIMULATION'>('DATA_CENTER_SIMULATION');
  const mutation = useUpdateSimulationControls();

  const handleApply = () => {
    mutation.mutate({
      simulated_load: load,
      ambient_temp_offset: offset,
      trigger_spike: load > 80,
      mode
    });
  };

  return (
    <div className="absolute bottom-4 left-4 z-10 w-80 rounded-lg border border-thervo-border bg-thervo-panel/90 p-4 text-thervo-text backdrop-blur-md">
      <div className="mb-4 flex items-center justify-between text-sm font-bold text-thervo-orange">
        <div className="flex items-center gap-2">
          <Settings2 size={14} /> CONTROLS
        </div>
        <select 
          value={mode}
          onChange={(e) => setMode(e.target.value as any)}
          className="bg-thervo-background text-thervo-text border border-thervo-border rounded px-2 py-1 text-xs outline-none"
        >
          <option value="DATA_CENTER_SIMULATION">SIMULATION MODE</option>
          <option value="LOCAL_LAPTOP">LOCAL LAPTOP</option>
        </select>
      </div>
      
      <div className="space-y-4 font-sans text-sm">
        <div>
          <label className="mb-1 flex justify-between">
            <span>Simulated Load</span>
            <span className="font-mono text-thervo-cool">{load}%</span>
          </label>
          <input 
            type="range" min="0" max="100" value={load} 
            onChange={(e) => setLoad(Number(e.target.value))}
            className="w-full accent-thervo-orange"
          />
        </div>

        <div>
          <label className="mb-1 flex justify-between">
            <span>Thermal Offset</span>
            <span className="font-mono text-thervo-cool">+{offset}°C</span>
          </label>
          <input 
            type="range" min="0" max="20" value={offset} 
            onChange={(e) => setOffset(Number(e.target.value))}
            className="w-full accent-thervo-orange"
          />
        </div>

        <button 
          onClick={handleApply}
          disabled={mutation.isPending}
          className="w-full rounded bg-thervo-border py-2 text-center font-bold hover:bg-thervo-cool/20 hover:text-thervo-cool disabled:opacity-50"
        >
          {mutation.isPending ? 'APPLYING...' : 'APPLY TUNING'}
        </button>
      </div>
    </div>
  );
}
