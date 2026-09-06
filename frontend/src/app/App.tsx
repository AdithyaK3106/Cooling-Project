import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { DataCenterScene } from '../three/DataCenterScene';

import { AnalyticsDashboard } from '../features/AnalyticsDashboard';
import { TelemetryRecorder } from '../services/TelemetryRecorder';

import { KeyPerformancePanel } from '../features/KeyPerformancePanel';
import { ContextualDetailsPanel } from '../features/ContextualDetailsPanel';
import { CompactBottomBar } from '../features/CompactBottomBar';
import { SceneControlBar } from '../features/SceneControlBar';

const queryClient = new QueryClient();

function CommandCenter() {
  return (
    <div className="absolute inset-0 overflow-hidden">
      {/* 3D Background */}
      <div className="absolute inset-0">
        <DataCenterScene />
      </div>

      {/* Floating UI Layer */}
      <div className="absolute inset-0 pointer-events-none p-6 flex flex-col justify-between">
        
        {/* Main 3-Column Layout */}
        <div className="flex-1 flex justify-between mt-12 pointer-events-none">
          {/* Left: Key Performance */}
          <div className="w-[320px] flex flex-col gap-4 pointer-events-auto">
            <KeyPerformancePanel />
          </div>

          {/* Center: 3D controls floating at top of center */}
          <div className="flex-1 flex justify-center items-start pointer-events-none">
             <SceneControlBar />
          </div>

          {/* Right: Contextual Details */}
          <div className="w-[320px] flex flex-col gap-4 pointer-events-auto items-end">
            <ContextualDetailsPanel />
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="h-20 mt-4 pointer-events-auto">
          <CompactBottomBar />
        </div>
      </div>
    </div>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <TelemetryRecorder />
        <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-[#05070a] text-gray-200">
          <div className="absolute left-6 top-6 z-20 flex gap-8 items-baseline pointer-events-auto">
            <div>
              <h1 className="text-2xl font-bold tracking-widest text-white">THERVO</h1>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-cyan-400">Command Center Online</p>
            </div>
            <nav className="flex gap-6 font-mono text-xs font-bold tracking-widest">
              <Link to="/" className="text-white">3D TWIN</Link>
              <Link to="/analytics" className="text-gray-500 hover:text-white transition-colors">ANALYTICS</Link>
            </nav>
          </div>
          
          <Routes>
            <Route path="/" element={<CommandCenter />} />
            <Route path="/analytics" element={<AnalyticsDashboard />} />
          </Routes>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
