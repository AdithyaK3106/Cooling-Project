import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useAnalyticsStore } from '../stores/analyticsStore';

export function AnalyticsDashboard() {
  const history = useAnalyticsStore((state) => state.history);

  return (
    <div className="flex h-full w-full flex-col bg-thervo-background p-8 pt-24 text-thervo-text">
      <h2 className="mb-6 text-2xl font-mono text-thervo-orange border-b border-thervo-border pb-2">Global Analytics</h2>
      
      <div className="grid grid-cols-2 gap-8 h-[400px]">
        {/* Temperature & CPU Chart */}
        <div className="rounded-lg border border-thervo-border bg-[#1C1F26] p-4">
          <h3 className="mb-4 text-sm font-bold text-thervo-cool">Average CPU Load & Temperature</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2D3342" />
              <XAxis dataKey="time" stroke="#6F9BA8" fontSize={12} />
              <YAxis stroke="#6F9BA8" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#1C1F26', borderColor: '#2D3342' }} />
              <Legend />
              <Line type="monotone" dataKey="avgCpu" name="Avg CPU %" stroke="#00ffff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="avgTemp" name="Avg Temp °C" stroke="#ff9900" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Score Chart */}
        <div className="rounded-lg border border-thervo-border bg-[#1C1F26] p-4">
          <h3 className="mb-4 text-sm font-bold text-thervo-orange">Peak Thermal Risk Score</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2D3342" />
              <XAxis dataKey="time" stroke="#6F9BA8" fontSize={12} />
              <YAxis stroke="#6F9BA8" fontSize={12} domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1C1F26', borderColor: '#2D3342' }} />
              <Legend />
              <Line type="stepAfter" dataKey="maxRisk" name="Max Risk (0-1)" stroke="#ff3333" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
