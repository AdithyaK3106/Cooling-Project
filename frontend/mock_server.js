import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

let load = 50;
let offset = 0;
let spike = false;
let currentMode = 'DATA_CENTER_SIMULATION';

app.get('/telemetry', (req, res) => {
  const time = Date.now() / 1000;
  const baseTemp = 40 + offset;
  
  const generateRack = (id, baseIndex) => {
    const oscillation = Math.sin(time * 0.5 + baseIndex) * 10; 
    const currentTemp = 40 + (load / 100) * 40 + offset + oscillation + (spike ? 20 : 0);
    const risk = Math.max(0, Math.min(1, (currentTemp - 40) / 60)); // 0 to 1
    return {
      id,
      telemetry: {
        cpu_util: Math.max(0, Math.min(100, load + (Math.sin(time + baseIndex) * 10))),
        gpu_util: Math.max(0, Math.min(100, load + (Math.cos(time + baseIndex) * 10))),
        cpu_temp: currentTemp
      },
      risk_score: risk,
      cooling: {
        target_rpm: load * 50,
        actual_rpm: load * 50 + Math.random() * 100 - 50,
        status: risk > 0.6 ? 'predictive intervention' : 'normal'
      }
    };
  };

  const racks = currentMode === 'LOCAL_LAPTOP'
    ? [generateRack('A07', 7)]
    : Array.from({ length: 25 }, (_, i) => generateRack(`A0${i + 1}`, i));

  // Generate thermal spread topology for 5x5 grid
  const topology = [];
  if (currentMode !== 'LOCAL_LAPTOP') {
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const id1 = row * 5 + col + 1;
        const rack1 = `A0${id1}`;
        
        // Right neighbor
        if (col < 4) {
          const id2 = row * 5 + col + 2;
          topology.push({ source: rack1, target: `A0${id2}`, weight: Math.random() * 0.8 + 0.2 });
        }
        // Bottom neighbor
        if (row < 4) {
          const id3 = (row + 1) * 5 + col + 1;
          topology.push({ source: rack1, target: `A0${id3}`, weight: Math.random() * 0.8 + 0.2 });
        }
      }
    }
  }

  res.json({
    operating_mode: currentMode,
    global_health: {
      status: spike ? 'critical' : 'healthy',
      issues: spike ? ['Thermal runaway predicted in zone A'] : []
    },
    events: [
      { time: new Date().toLocaleTimeString(), message: 'System running normally' },
      ...(spike ? [{ time: new Date().toLocaleTimeString(), message: 'Thermal spike detected!' }] : [])
    ],
    racks,
    topology
  });
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
