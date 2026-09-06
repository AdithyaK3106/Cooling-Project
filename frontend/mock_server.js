import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

const NUM_RACKS = 25;
let currentMode = 'DATA_CENTER_SIMULATION';
let load = 50;
let noise = 12;
let spikeTarget = -1;
let epoch = 0;
let events = [{ time: new Date().toLocaleTimeString(), message: 'System initialized' }];

const ZONES = Array.from({length: 25}, (_, i) => {
  if (i < 5) return 'A';
  if (i < 10) return 'B';
  if (i < 15) return 'C';
  if (i < 20) return 'D';
  return 'E';
});

// Generate 5x5 topology
const GNN_EDGES = [];
for (let row = 0; row < 5; row++) {
  for (let col = 0; col < 5; col++) {
    const id1 = row * 5 + col;
    if (col < 4) GNN_EDGES.push([id1, row * 5 + col + 1]);
    if (row < 4) GNN_EDGES.push([id1, (row + 1) * 5 + col]);
  }
}

function buildXGBModel() {
  const trees = [];
  const treeTemplates = [
    [0, 0.30, 0.02, 0.18], [1, 0.28, 0.02, 0.16], [5, 0.25, 0.02, 0.15],
    [0, 0.45, 0.04, 0.28], [1, 0.42, 0.03, 0.26], [5, 0.38, 0.03, 0.22],
    [0, 0.65, 0.05, 0.35], [1, 0.62, 0.04, 0.32], [5, 0.55, 0.04, 0.28],
    [0, 0.80, 0.06, 0.42], [1, 0.78, 0.06, 0.38], [2, 0.50, 0.02, 0.14],
  ];
  for (let t = 0; t < 30; t++) {
    const base = treeTemplates[t % treeTemplates.length];
    const jitter = 1 + (Math.sin(t * 1.7) * 0.1);
    trees.push([base[0], base[1] * jitter, base[2], base[3] * jitter]);
  }
  return trees;
}
const XGB_TREES = buildXGBModel();

function xgbPredict(features) {
  const LEARNING_RATE = 0.085;
  let score = 0.10;
  for (const [fi, thresh, leftVal, rightVal] of XGB_TREES) {
    score += LEARNING_RATE * (features[fi] > thresh ? rightVal : leftVal);
  }
  return Math.min(1.0, Math.max(0.05, score));
}

let racks = Array.from({length: NUM_RACKS}, (_, i) => ({
  id: `A0${i+1}`,
  zone: ZONES[i],
  cpu: 0, gpu: 0, memory: 0, diskIO: 0, network: 0,
  gnnEmbed: 0, riskScore: 0, xgbPred: 0,
  coolingActive: false, overrideEnabled: false, spikeBonus: 0
}));

function addLog(msg) {
  events.unshift({ time: new Date().toLocaleTimeString(), message: msg });
  if (events.length > 5) events.pop();
}

function generateSyntheticWorkload(rackIdx, epoch, baseLoad, noiseFactor) {
  let traceCpu = 0.28, traceGpu = 0.32, traceMem = 0.35, traceDisk = 0.20, traceNet = 0.25;
  const loadScale = baseLoad / 0.35;
  const rackBias = [1.25, 0.88, 1.12, 0.78, 1.35][rackIdx % 5] || 1.0;
  const n = () => (Math.random() - 0.5) * noiseFactor * 2.0;

  let cpu = Math.min(99, Math.max(10, traceCpu * 100 * loadScale * rackBias + n() * 6));
  let gpu = Math.min(99, Math.max(5, traceGpu * 100 * loadScale * rackBias + n() * 6));
  let memory = Math.min(99, Math.max(18, traceMem * 100 * (0.85 + loadScale * 0.15) + n() * 4));
  let diskIO = Math.min(99, Math.max(3, traceDisk * 100 * loadScale + n() * 5));
  let network = Math.min(99, Math.max(5, traceNet * 100 * loadScale + n() * 6));

  const activeRack = racks[rackIdx];
  if (activeRack && activeRack.spikeBonus > 0) {
    cpu = Math.min(99, cpu + activeRack.spikeBonus);
    gpu = Math.min(99, gpu + activeRack.spikeBonus * 0.9);
    activeRack.spikeBonus *= 0.90;
    if (activeRack.spikeBonus < 0.5) activeRack.spikeBonus = 0;
  }
  return { cpu, gpu, memory, diskIO, network };
}

function computeGNNEmbeddings(rackFeatures) {
  const embeddings = [];
  const adjList = Array.from({length: NUM_RACKS}, () => []);
  GNN_EDGES.forEach(([a,b]) => { adjList[a].push(b); adjList[b].push(a); });

  for (let i = 0; i < NUM_RACKS; i++) {
    const self = rackFeatures[i];
    const selfCooling = racks[i] ? (racks[i].coolingActive || racks[i].overrideEnabled) : false;
    const selfCoolFactor = selfCooling ? 0.45 : 1.0;

    const neighbors = adjList[i].map(j => {
      const feat = rackFeatures[j];
      const jCooling = racks[j] ? (racks[j].coolingActive || racks[j].overrideEnabled) : false;
      return { heat: (feat.cpu * 0.6 + feat.gpu * 0.4) * (jCooling ? 0.45 : 1.0) };
    });

    const neighborHeat = neighbors.length > 0 ? neighbors.reduce((s, n) => s + n.heat, 0) / neighbors.length : 0;
    const selfHeat = (self.cpu * 0.6 + self.gpu * 0.4) * selfCoolFactor;
    embeddings.push(parseFloat(((selfHeat * 0.7 + neighborHeat * 0.3) / 100).toFixed(4)));
  }
  return embeddings;
}

// Tick loop
setInterval(() => {
  epoch++;
  const rawFeatures = racks.map((_, i) => generateSyntheticWorkload(i, epoch, load/100, noise/100));
  const gnnEmbeds = computeGNNEmbeddings(rawFeatures);

  racks.forEach((rack, i) => {
    const f = rawFeatures[i];
    rack.cpu = f.cpu; rack.gpu = f.gpu; rack.memory = f.memory; rack.diskIO = f.diskIO; rack.network = f.network;
    rack.gnnEmbed = gnnEmbeds[i];

    const featVec = [f.cpu/100, f.gpu/100, f.memory/100, f.diskIO/100, f.network/100, gnnEmbeds[i]];
    rack.xgbPred = xgbPredict(featVec);
    const rawCompositeRisk = rack.xgbPred * 0.75 + rack.gnnEmbed * 0.25;

    // Autonomous cooling trigger
    if (!rack.coolingActive) {
      if (rawCompositeRisk >= 0.58 || rack.riskScore >= 0.58) {
        rack.coolingActive = true;
        addLog(`⚠ AUTO COOLING DEPLOYED: ${rack.id} risk=${(rawCompositeRisk*100).toFixed(0)}%`);
      }
    } else {
      if (rack.riskScore <= 0.45 && rawCompositeRisk <= 0.45 && !rack.overrideEnabled) {
        rack.coolingActive = false;
        addLog(`✓ ${rack.id} thermal state normalized — cooling standby`);
      }
    }

    const isCooled = rack.coolingActive || rack.overrideEnabled;
    const targetRisk = isCooled ? (rawCompositeRisk * 0.45) : rawCompositeRisk;
    const lerpFactor = isCooled ? 0.38 : 0.20;

    if (rack.riskScore === 0) rack.riskScore = parseFloat(targetRisk.toFixed(4));
    else rack.riskScore = parseFloat((rack.riskScore * (1 - lerpFactor) + targetRisk * lerpFactor).toFixed(4));
  });
}, 250);

app.get('/telemetry', (req, res) => {
  const topology = GNN_EDGES.map(([a,b]) => ({
    source: `A0${a+1}`, target: `A0${b+1}`,
    weight: ((racks[a].riskScore + racks[b].riskScore) / 2) > 0.5 ? 0.9 : 0.3
  }));

  const responseRacks = currentMode === 'LOCAL_LAPTOP' ? [racks[6]] : racks;
  const hotZones = racks.filter(r => r.riskScore > 0.55).length;

  res.json({
    operating_mode: currentMode,
    global_health: {
      status: hotZones > 3 ? 'critical' : 'healthy',
      issues: hotZones > 3 ? [`${hotZones} zones critical`] : []
    },
    events,
    topology,
    racks: responseRacks.map(r => ({
      id: r.id,
      telemetry: { cpu_util: r.cpu, gpu_util: r.gpu, cpu_temp: r.riskScore * 100 },
      risk_score: r.riskScore,
      cooling: { status: r.coolingActive ? 'predictive intervention' : 'normal' },
      ai_insights: { gnn_embed: r.gnnEmbed, xgb_pred: r.xgbPred, zone: r.zone }
    }))
  });
});

app.post('/simulation/controls', (req, res) => {
  const p = req.body;
  if (p.simulated_load !== undefined) load = p.simulated_load;
  if (p.trigger_spike) {
    const target = Math.floor(Math.random() * NUM_RACKS);
    racks[target].spikeBonus = 60;
    addLog(`⚡ THERMAL SPIKE INJECTED into ${racks[target].id}`);
  }
  res.json({ success: true });
});

app.listen(8000, '0.0.0.0', () => console.log('Mock server running on port 8000'));
