import { useTelemetry } from '../services/telemetryApi';
import { Terminal } from 'lucide-react';

export function EventsTab() {
  const { data: telemetry } = useTelemetry();

  return (
    <div className="flex h-full w-full flex-col p-6 overflow-hidden">
      <div className="flex-1 rounded-xl border border-white/10 bg-[#0B0E14]/80 backdrop-blur shadow-2xl flex flex-col overflow-hidden">
        
        <div className="p-6 border-b border-white/10 flex justify-between items-center bg-white/5">
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
              <Terminal size={20} className="text-cyan-400" />
              Operational Events Audit Stream
            </h2>
            <p className="text-sm text-gray-400 mt-1">Live audit log of all system interventions and status changes</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
            <span className="text-xs font-mono font-bold text-green-400">STREAMING</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 font-mono text-sm bg-black/40">
          {telemetry?.events?.length ? (
            <div className="flex flex-col gap-3">
              {telemetry.events.map((evt, i) => {
                const isWarning = evt.message.includes('⚠') || evt.message.includes('SPIKE');
                const isSuccess = evt.message.includes('✓');
                
                const textColor = isWarning ? 'text-yellow-400' : isSuccess ? 'text-green-400' : 'text-gray-300';
                
                return (
                  <div key={i} className="flex gap-4 border-b border-white/5 pb-3">
                    <span className="text-cyan-500 opacity-70 w-24 shrink-0">{evt.time}</span>
                    <span className={textColor}>{evt.message}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-gray-500 italic">No events recorded yet...</div>
          )}
        </div>

      </div>
    </div>
  );
}
