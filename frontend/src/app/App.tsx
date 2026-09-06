
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex h-screen w-screen flex-col items-center justify-center bg-thervo-background text-thervo-text">
        <h1 className="text-3xl font-mono text-thervo-cool">THERVO</h1>
        <p className="mt-4 font-sans">Command Center Initializing...</p>
      </div>
    </QueryClientProvider>
  );
}
