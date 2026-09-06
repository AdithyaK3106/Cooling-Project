import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { TopNav } from '../features/TopNav';
import { StatusStrip } from '../features/StatusStrip';
import { OverviewTab } from '../pages/OverviewTab';
import { ThermalMapTab } from '../pages/ThermalMapTab';

import { RacksTab } from '../pages/RacksTab';
import { PredictionsTab } from '../pages/PredictionsTab';
import { EventsTab } from '../pages/EventsTab';

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="flex h-screen flex-col bg-[#0B0E14] text-white overflow-hidden font-sans">
          <TopNav />
          <StatusStrip />
          
          <main className="flex-1 overflow-auto relative">
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              <Route path="/overview" element={<OverviewTab />} />
              <Route path="/thermal-map" element={<ThermalMapTab />} />
              <Route path="/racks" element={<RacksTab />} />
              <Route path="/predictions" element={<PredictionsTab />} />
              <Route path="/events" element={<EventsTab />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  );
}
