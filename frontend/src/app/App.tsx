import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { TopNav } from '../features/TopNav';
import { OverviewTab } from '../pages/OverviewTab';
import { ThermalMapTab } from '../pages/ThermalMapTab';
import { RacksTab } from '../pages/RacksTab';
import { PredictionsTab } from '../pages/PredictionsTab';
import { EventsTab } from '../pages/EventsTab';
import { DualNodeDemo } from '../pages/DualNodeDemo';
import { LocalPipelineTab } from '../pages/LocalPipelineTab';
import { LandingPage } from '../pages/LandingPage';

const queryClient = new QueryClient();

function AppContent() {
  const location = useLocation();
  const isLanding = location.pathname === '/';
  const mode = localStorage.getItem('thervo_mode') || 'LOCAL';
  const enterRoute = mode === 'SIMULATED' ? '/overview' : '/dual-node';

  if (isLanding) {
    return (
      <LandingPage
        videoSrc="/use_this.mp4"
        enterTargetRoute={enterRoute}
        brandName="THERVO"
      />
    );
  }

  return (
    <div className="flex h-screen flex-col bg-[#0B0E14] text-white overflow-hidden font-sans">
      <TopNav />
      
      <main className="flex-1 overflow-auto relative">
        <Routes>
          <Route path="/dual-node" element={<DualNodeDemo />} />
          <Route path="/pipeline" element={<LocalPipelineTab />} />
          <Route path="/overview" element={<OverviewTab />} />
          <Route path="/thermal-map" element={<ThermalMapTab />} />
          <Route path="/racks" element={<RacksTab />} />
          <Route path="/predictions" element={<PredictionsTab />} />
          <Route path="/events" element={<EventsTab />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppContent />
      </Router>
    </QueryClientProvider>
  );
}
