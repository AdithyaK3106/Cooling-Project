import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DataCenterScene } from '../three/DataCenterScene';
import { EventStream } from '../components/events/EventStream';
import { ScenarioControlPanel } from '../components/controls/ScenarioControlPanel';
import { RackInspector } from '../components/rack/RackInspector';

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-thervo-background text-thervo-text">
        <div className="absolute left-4 top-4 z-10">
          <h1 className="text-3xl font-mono text-thervo-cool">THERVO</h1>
          <p className="mt-2 font-sans text-sm text-thervo-text opacity-70">Command Center Online</p>
        </div>
        
        <DataCenterScene />
        <RackInspector />
        <ScenarioControlPanel />
        <EventStream />
      </div>
    </QueryClientProvider>
  );
}
