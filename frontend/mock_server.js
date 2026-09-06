import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

let load = 50;
let offset = 0;
let spike = false;
let time = 0;

app.get('/telemetry', (req, res) => {
  time += 0.1;
  const baseTemp = 40 + (load / 100) * 40 + offset + (spike ? 20 : 0);
  
  const telemetry = {
    operating_mode: 'DATA_CENTER_SIMULATION',
    global_health: { status: 'OK', issues: [] },
    events: [
      { time: new Date().toLocaleTimeString(), message: 'System running normally' },
      ...(spike ? [{ time: new Date().toLocaleTimeString(), message: 'Thermal spike detected!' }] : [])
    ],
    racks: Array.from({ length: 360 }, (_, i) => {
      // Use the simulated load and offset applied by the user!
      // Add a slight sine wave to make it feel alive, but respect the base load!
      const oscillation = Math.sin(time * 0.5 + i) * 10; 
      const currentTemp = 40 + (load / 100) * 40 + offset + oscillation + (spike ? 20 : 0);
      
      const risk = Math.max(0, Math.min(1, (currentTemp - 40) / 60)); // 0 to 1
      
      return {
        id: `A0${i + 1}`,
        telemetry: {
          cpu_util: Math.max(0, Math.min(100, load + (Math.sin(time + i) * 10))),
          gpu_util: Math.max(0, Math.min(100, load + (Math.cos(time + i) * 10))),
          cpu_temp: currentTemp
        },
        risk_score: risk,
        cooling: {
          target_rpm: load * 50,
          actual_rpm: load * 50 + Math.random() * 100 - 50,
          status: risk > 0.6 ? 'predictive intervention' : 'normal'
        }
      };
    }),
    topology: [
      { source: 'A01', target: 'A02', weight: 0.8 },
      { source: 'A02', target: 'A03', weight: 0.5 },
      { source: 'A03', target: 'A04', weight: 0.9 },
      { source: 'A04', target: 'A05', weight: 0.6 },
      { source: 'A05', target: 'A06', weight: 0.7 },
      { source: 'A06', target: 'A07', weight: 0.4 },
    ]
  };
  
  res.json(telemetry);
});

app.post('/simulation/controls', (req, res) => {
  const params = req.body;
  load = params.simulated_load;
  offset = params.ambient_temp_offset;
  spike = params.trigger_spike;
  console.log('Received controls:', params);
  res.json({ success: true });
});

app.listen(8000, '0.0.0.0', () => {
  console.log('Mock server running on port 8000');
});
