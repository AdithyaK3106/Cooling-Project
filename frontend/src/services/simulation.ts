const NUM_RACKS = 25;
let currentMode = 'DATA_CENTER_SIMULATION';
let load = 50;
let noise = 12;
let dissipationRate = 1.0;
let thermalMultiplier = 1.0;
let simSpeedMs = 500;
let isPaused = false;
let epoch = 0;
let totalPreds = 0;
let alertCount = 0;
let eventsList: Array<{ time: string; source: string; category: string; message: string }> = [];

let configListeners: Array<() => void> = [];

function notifyConfigListeners() {
  configListeners.forEach(cb => cb());
}

export function subscribeSimulationConfig(callback: () => void) {
  configListeners.push(callback);
  return () => {
    configListeners = configListeners.filter(cb => cb !== callback);
  };
}

export function getSimulationConfig() {
  return {
    load,
    noise,
    dissipationRate,
    thermalMultiplier,
    simSpeedMs,
    isPaused
  };
}

export function setSimulationSpeed(speedMs: number) {
  simSpeedMs = Math.max(50, Math.min(3000, speedMs));
  addEvent(`Simulation speed updated to ${simSpeedMs}ms / tick (${(1000/simSpeedMs).toFixed(1)} Hz)`, 'ACTION', 'CONTROLS');
  notifyConfigListeners();
}

export function togglePauseSimulation() {
  isPaused = !isPaused;
  addEvent(isPaused ? 'Simulation PAUSED' : 'Simulation RESUMED', 'ACTION', 'CONTROLS');
  notifyConfigListeners();
}

export function stepSimulation() {
  const wasPaused = isPaused;
  isPaused = false;
  tickSimulation();
  isPaused = wasPaused;
  addEvent(`Stepped forward +1 tick (Epoch ${epoch})`, 'ACTION', 'CONTROLS');
  notifyConfigListeners();
}

export function setSimulationParams(newLoad: number, newNoise: number, newDissipation = 1.0, newThermalMult = 1.0) {
  load = Math.max(10, Math.min(100, newLoad));
  noise = Math.max(0, Math.min(50, newNoise));
  dissipationRate = Math.max(0.2, Math.min(3.0, newDissipation));
  thermalMultiplier = Math.max(0.5, Math.min(3.0, newThermalMult));
  addEvent(`Physics tuned: Load ${load}%, Cool Rate ${dissipationRate.toFixed(1)}x, Noise ${noise}%`, 'ACTION', 'CONTROLS');
  notifyConfigListeners();
}

export function injectGlobalSpike() {
  racks.forEach(r => {
    r.spikeBonus += 65;
  });
  alertCount += NUM_RACKS;
  addEvent('EMERGENCY: Global thermal heat spike injected across ALL 25 racks!', 'WARN', 'SIMULATION');
  notifyConfigListeners();
}

const ZONES = Array.from({length: NUM_RACKS}, (_, i) => {
  if (i < 5) return 'A';
  if (i < 10) return 'B';
  if (i < 15) return 'C';
  if (i < 20) return 'D';
  return 'E';
});

const GNN_EDGES: number[][] = [];
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

function xgbPredict(features: number[]) {
  const LEARNING_RATE = 0.085;
  let score = 0.10;
  for (const [fi, thresh, leftVal, rightVal] of XGB_TREES) {
    score += LEARNING_RATE * (features[fi] > thresh ? rightVal : leftVal);
  }
  return Math.min(1.0, Math.max(0.05, score));
}

let racks = Array.from({length: NUM_RACKS}, (_, i) => ({
  id: `A0${i+1 < 10 ? '0' + (i+1) : (i+1)}`,
  zone: ZONES[i],
  cpu: 0, gpu: 0, memory: 0, diskIO: 0, network: 0,
  gnnEmbed: 0, riskScore: 0, xgbPred: 0,
  coolingActive: false, overrideEnabled: false, spikeBonus: 0
}));

function getTimeString() {
  const now = new Date();
  return `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
}

function addEvent(message: string, category: string = 'ACTION', source: string = 'SIMULATION') {
  eventsList.unshift({
    time: getTimeString(),
    source,
    category,
    message
  });
  if (eventsList.length > 20) eventsList.pop();
}

function generateSyntheticWorkload(rackIdx: number, baseLoad: number, noiseFactor: number) {
  let traceCpu = 0.28, traceGpu = 0.32, traceMem = 0.35, traceDisk = 0.20, traceNet = 0.25;
  const loadScale = baseLoad / 0.35;
  const rackBias = [1.25, 0.88, 1.12, 0.78, 1.35][rackIdx % 5] || 1.0;

  // Workload math depends strictly on simulation epoch tick count so pausing freezes time
  const t = epoch * 0.15;
  const sine1 = Math.sin(t * 0.9 + rackIdx * 1.3);
  const sine2 = Math.cos(t * 1.5 + rackIdx * 0.7);
  const n = () => (Math.sin(epoch * 3.1 + rackIdx * 2.7) * noiseFactor * 2.0);

  let cpu = Math.min(99, Math.max(10, traceCpu * 100 * loadScale * rackBias + sine1 * 14 + n() * 6));
  let gpu = Math.min(99, Math.max(5, traceGpu * 100 * loadScale * rackBias + sine2 * 16 + n() * 6));
  let memory = Math.min(99, Math.max(18, traceMem * 100 * (0.85 + loadScale * 0.15) + Math.sin(t * 0.4 + rackIdx) * 6 + n() * 3));
  let diskIO = Math.min(99, Math.max(3, traceDisk * 100 * loadScale * (1 + Math.abs(sine2) * 0.6) + n() * 8));
  let network = Math.min(99, Math.max(5, traceNet * 100 * loadScale * (1 + Math.abs(sine1) * 0.7) + n() * 10));

  const activeRack = racks[rackIdx];
  if (activeRack && activeRack.spikeBonus > 0) {
    cpu = Math.min(99, cpu + activeRack.spikeBonus);
    gpu = Math.min(99, gpu + activeRack.spikeBonus * 0.9);
    activeRack.spikeBonus *= 0.88;
    if (activeRack.spikeBonus < 0.5) activeRack.spikeBonus = 0;
  }
  return { cpu, gpu, memory, diskIO, network };
}

function computeGNNEmbeddings(rackFeatures: any[]) {
  const embeddings: number[] = [];
  const adjList: number[][] = Array.from({length: NUM_RACKS}, () => []);
  GNN_EDGES.forEach(([a,b]) => { adjList[a].push(b); adjList[b].push(a); });

  for (let i = 0; i < NUM_RACKS; i++) {
    const self = rackFeatures[i];
    const selfCooling = racks[i] ? (racks[i].coolingActive || racks[i].overrideEnabled) : false;
    const selfCoolFactor = selfCooling ? (0.45 / Math.max(0.2, dissipationRate)) : 1.0;

    const neighbors = adjList[i].map(j => {
      const feat = rackFeatures[j];
      const jCooling = racks[j] ? (racks[j].coolingActive || racks[j].overrideEnabled) : false;
      return { heat: (feat.cpu * 0.6 + feat.gpu * 0.4) * (jCooling ? (0.45 / Math.max(0.2, dissipationRate)) : 1.0) };
    });

    const neighborHeat = neighbors.length > 0 ? neighbors.reduce((s, n) => s + n.heat, 0) / neighbors.length : 0;
    const selfHeat = (self.cpu * 0.6 + self.gpu * 0.4) * selfCoolFactor;
    embeddings.push(parseFloat(((selfHeat * 0.7 + neighborHeat * 0.3) / 100).toFixed(4)));
  }
  return embeddings;
}

export function tickSimulation() {
  if (isPaused) return;

  epoch++;
  totalPreds += NUM_RACKS;
  const rawFeatures = racks.map((_, i) => generateSyntheticWorkload(i, (load / 100) * thermalMultiplier, noise / 100));
  const gnnEmbeds = computeGNNEmbeddings(rawFeatures);

  racks.forEach((rack, i) => {
    const f = rawFeatures[i];
    rack.cpu = f.cpu; rack.gpu = f.gpu; rack.memory = f.memory; rack.diskIO = f.diskIO; rack.network = f.network;
    rack.gnnEmbed = gnnEmbeds[i];

    const featVec = [f.cpu/100, f.gpu/100, f.memory/100, f.diskIO/100, f.network/100, gnnEmbeds[i]];
    rack.xgbPred = xgbPredict(featVec);
    const rawCompositeRisk = rack.xgbPred * 0.75 + rack.gnnEmbed * 0.25;

    if (!rack.coolingActive) {
      if (rawCompositeRisk >= 0.55 || rack.riskScore >= 0.55) {
        rack.coolingActive = true;
        alertCount++;
        addEvent(`Auto predictive cooling engaged on ${rack.id}`, 'WARN', 'GNN_AI');
      }
    } else {
      if (rack.riskScore <= 0.42 && rawCompositeRisk <= 0.42 && !rack.overrideEnabled) {
        rack.coolingActive = false;
        addEvent(`Thermal equilibrium reached on ${rack.id}`, 'HEALTHY', 'THERMAL');
      }
    }

    const isCooled = rack.coolingActive || rack.overrideEnabled;
    const coolFactor = Math.min(0.85, 0.40 / Math.max(0.2, dissipationRate));
    const targetRisk = isCooled ? (rawCompositeRisk * coolFactor) : rawCompositeRisk;
    
    // Smooth responsive interpolation based on tick speed
    const stepRatio = Math.min(0.75, Math.max(0.15, 0.35 * (simSpeedMs / 250)));
    rack.riskScore = parseFloat((rack.riskScore * (1 - stepRatio) + targetRisk * stepRatio).toFixed(4));
  });
}

export function injectRandomSpike() {
  const randomRack = racks[Math.floor(Math.random() * racks.length)];
  if (randomRack) {
    randomRack.spikeBonus += 70;
    addEvent(`Workload spike injected into ${randomRack.id}`, 'WARN', 'SIMULATION');
  }
}

export function injectRackSpike(rackId: string) {
  const rack = racks.find(r => r.id === rackId);
  if (rack) {
    rack.spikeBonus += 70;
    addEvent(`Thermal load spike applied to ${rack.id}`, 'WARN', 'SIMULATION');
  }
}

export function toggleRackOverride(rackId: string) {
  const rack = racks.find(r => r.id === rackId);
  if (rack) {
    rack.overrideEnabled = !rack.overrideEnabled;
    const statusStr = rack.overrideEnabled ? 'OVERRIDE ENABLED' : 'OVERRIDE RELEASED';
    addEvent(`Manual cooling ${statusStr} for ${rack.id}`, 'ACTION', 'USER');
  }
}

export function resetSimulation() {
  epoch = 0;
  totalPreds = 0;
  alertCount = 0;
  load = 50;
  noise = 12;
  eventsList = [];
  racks.forEach(r => {
    r.cpu = 0; r.gpu = 0; r.memory = 0; r.diskIO = 0; r.network = 0;
    r.gnnEmbed = 0; r.riskScore = 0; r.xgbPred = 0;
    r.coolingActive = false; r.overrideEnabled = false; r.spikeBonus = 0;
  });
  addEvent('Simulation state reset to baseline', 'HEALTHY', 'SYSTEM');
}

export function getSimulatedTelemetry(): any {
  const topology = GNN_EDGES.map(([a,b]) => ({
    source: racks[a].id, target: racks[b].id,
    weight: ((racks[a].riskScore + racks[b].riskScore) / 2) > 0.5 ? 0.9 : 0.3
  }));

  const hotZones = racks.filter(r => r.riskScore > 0.55).length;
  const modelAccuracy = Math.min(99.5, 91.0 + Math.sin(epoch * 0.07) * 1.8 + Math.min(epoch * 0.02, 4)).toFixed(1);

  if (eventsList.length === 0) {
    addEvent('Simulated Datacenter Environment Active', 'HEALTHY', 'CORE');
  }

  return {
    operating_mode: currentMode,
    global_health: {
      status: hotZones > 3 ? 'critical' : 'healthy',
      issues: hotZones > 3 ? [`${hotZones} zones critical`] : []
    },
    model_stats: {
      accuracy: parseFloat(modelAccuracy),
      total_predictions: totalPreds,
      active_alerts: alertCount
    },
    events: eventsList,
    topology,
    racks: racks.map(r => ({
      id: r.id,
      telemetry: {
        cpu_util: parseFloat(r.cpu.toFixed(1)),
        gpu_util: parseFloat(r.gpu.toFixed(1)),
        mem_util: parseFloat(r.memory.toFixed(1)),
        disk_io: parseFloat(r.diskIO.toFixed(1)),
        network_io: parseFloat(r.network.toFixed(1)),
        cpu_temp: parseFloat((35 + r.riskScore * 50).toFixed(1)),
        gpu_temp: parseFloat((38 + r.riskScore * 48).toFixed(1)),
        power_draw: parseFloat((100 + r.cpu * 3 + r.gpu * 4).toFixed(1))
      },
      risk_score: r.riskScore,
      cooling: {
        status: (r.coolingActive || r.overrideEnabled) ? 'predictive intervention' : 'normal',
        override: r.overrideEnabled,
        actual_rpm: Math.round(1000 + r.riskScore * 3500)
      },
      ai_insights: { gnn_embed: r.gnnEmbed, xgb_pred: r.xgbPred, zone: r.zone }
    }))
  };
}
