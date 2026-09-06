import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { DataCenterScene } from '../three/DataCenterScene';
import { EventStream } from '../components/events/EventStream';
import { ScenarioControlPanel } from '../components/controls/ScenarioControlPanel';
import { RackInspector } from '../components/rack/RackInspector';
import { AnalyticsDashboard } from '../features/AnalyticsDashboard';
import { TelemetryRecorder } from '../services/TelemetryRecorder';

const queryClient = new QueryClient();

function CommandCenter() {
  return (
    <>
      <DataCenterScene />
      <RackInspector />
      <ScenarioControlPanel />
      <EventStream />
    </>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <TelemetryRecorder />
        <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-thervo-background text-thervo-text">
          <div className="absolute left-4 top-4 z-20 flex gap-6 items-baseline">
            <div>
              <h1 className="text-3xl font-mono text-thervo-cool">THERVO</h1>
              <p className="mt-1 font-sans text-xs text-thervo-text opacity-70">Command Center Online</p>
            </div>
            <nav className="flex gap-4 font-mono text-sm">
              <Link to="/" className="text-thervo-cool hover:text-white">3D TWIN</Link>
              <Link to="/analytics" className="text-thervo-orange hover:text-white">ANALYTICS</Link>
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
