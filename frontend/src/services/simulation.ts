
const NUM_RACKS = 25;
let currentMode = 'DATA_CENTER_SIMULATION';
let load = 50;
let noise = 12;
let epoch = 0;
let totalPreds = 0;
let alertCount = 0;

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
  id: `A0${i+1}`,
  zone: ZONES[i],
  cpu: 0, gpu: 0, memory: 0, diskIO: 0, network: 0,
  gnnEmbed: 0, riskScore: 0, xgbPred: 0,
  coolingActive: false, overrideEnabled: false, spikeBonus: 0,
  manualDisengaged: false,
  coolingProgress: 0.0
}));

function generateSyntheticWorkload(rackIdx: number, _epoch: number, baseLoad: number, noiseFactor: number) {
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

function computeGNNEmbeddings(rackFeatures: any[]) {
  const embeddings: number[] = [];
  const adjList: number[][] = Array.from({length: NUM_RACKS}, () => []);
  GNN_EDGES.forEach(([a,b]) => { adjList[a].push(b); adjList[b].push(a); });

  for (let i = 0; i < NUM_RACKS; i++) {
    const self = rackFeatures[i];
    const selfCooling = racks[i] ? (racks[i].coolingActive || racks[i].overrideEnabled) : false;
    const progress = (racks[i] as any)?.coolingProgress ?? (selfCooling ? 1.0 : 0.0);
    const selfCoolFactor = 1.0 - (1.0 - 0.45) * progress;

    const neighbors = adjList[i].map(j => {
      const feat = rackFeatures[j];
      const jCooling = racks[j] ? (racks[j].coolingActive || racks[j].overrideEnabled) : false;
      const jProgress = (racks[j] as any)?.coolingProgress ?? (jCooling ? 1.0 : 0.0);
      const jCoolFactor = 1.0 - (1.0 - 0.45) * jProgress;
      return { heat: (feat.cpu * 0.6 + feat.gpu * 0.4) * jCoolFactor };
    });

    const neighborHeat = neighbors.length > 0 ? neighbors.reduce((s, n) => s + n.heat, 0) / neighbors.length : 0;
    const selfHeat = (self.cpu * 0.6 + self.gpu * 0.4) * selfCoolFactor;
    embeddings.push(parseFloat(((selfHeat * 0.7 + neighborHeat * 0.3) / 100).toFixed(4)));
  }
  return embeddings;
}

export function tickSimulation() {
  epoch++;
  totalPreds += NUM_RACKS;
  const rawFeatures = racks.map((_, i) => generateSyntheticWorkload(i, epoch, load/100, noise/100));
  const gnnEmbeds = computeGNNEmbeddings(rawFeatures);

  racks.forEach((rack, i) => {
    const f = rawFeatures[i];
    rack.cpu = f.cpu; rack.gpu = f.gpu; rack.memory = f.memory; rack.diskIO = f.diskIO; rack.network = f.network;
    rack.gnnEmbed = gnnEmbeds[i];

    const featVec = [f.cpu/100, f.gpu/100, f.memory/100, f.diskIO/100, f.network/100, gnnEmbeds[i]];
    rack.xgbPred = xgbPredict(featVec);
    const rawCompositeRisk = rack.xgbPred * 0.75 + rack.gnnEmbed * 0.25;

    if (!rack.coolingActive) {
      if ((rawCompositeRisk >= 0.58 || rack.riskScore >= 0.58) && !rack.manualDisengaged) {
        rack.coolingActive = true;
        alertCount++;
      }
    } else {
      if (rack.riskScore <= 0.45 && rawCompositeRisk <= 0.45 && !rack.overrideEnabled) {
        rack.coolingActive = false;
      }
    }

    const isCooled = rack.coolingActive || rack.overrideEnabled;
    if (isCooled) {
      if ((rack as any).coolingProgress === undefined) (rack as any).coolingProgress = 0.0;
      (rack as any).coolingProgress = Math.min(1.0, (rack as any).coolingProgress + 0.025);
    } else {
      if ((rack as any).coolingProgress !== undefined && (rack as any).coolingProgress > 0) {
        (rack as any).coolingProgress = Math.max(0.0, (rack as any).coolingProgress - 0.10);
      } else {
        (rack as any).coolingProgress = 0.0;
      }
    }

    const progress = (rack as any).coolingProgress ?? (isCooled ? 1.0 : 0.0);
    const coolRatio = 1.0 - (1.0 - 0.45) * progress;
    const targetRisk = rawCompositeRisk * coolRatio;
    const lerpFactor = 0.08;

    if (rack.riskScore === 0) rack.riskScore = parseFloat(targetRisk.toFixed(4));
    else rack.riskScore = parseFloat((rack.riskScore * (1 - lerpFactor) + targetRisk * lerpFactor).toFixed(4));
  });
}

export function getSimulatedTelemetry(): any {
  const topology = GNN_EDGES.map(([a,b]) => ({
    source: `A0${a+1}`, target: `A0${b+1}`,
    weight: ((racks[a].riskScore + racks[b].riskScore) / 2) > 0.5 ? 0.9 : 0.3
  }));

  const hotZones = racks.filter(r => r.riskScore > 0.55).length;
  const modelAccuracy = Math.min(99.5, 91.0 + Math.sin(epoch * 0.07) * 1.8 + Math.min(epoch * 0.02, 4)).toFixed(1);

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
    events: [],
    topology,
    racks: racks.map(r => ({
      id: r.id,
      telemetry: { cpu_util: r.cpu, gpu_util: r.gpu, cpu_temp: r.riskScore * 100 },
      risk_score: r.riskScore,
      cooling: { status: (r.coolingActive || r.overrideEnabled) ? 'predictive intervention' : 'normal' },
      coolingActive: r.coolingActive,
      overrideEnabled: r.overrideEnabled,
      ai_insights: { gnn_embed: r.gnnEmbed, xgb_pred: r.xgbPred, zone: r.zone }
    }))
  };
}

export function setRackCooling(rackId: string, enabled: boolean) {
  const numMatch = (rackId || '').match(/\d+/);
  const rackNum = numMatch ? parseInt(numMatch[0], 10) : -1;
  const target = racks.find(r => 
    r.id === rackId || 
    r.id === `A0${rackNum}` || 
    r.id === `A${rackNum}` ||
    (rackNum > 0 && parseInt(r.id.replace(/\D+/g, ''), 10) === rackNum)
  );
  if (target) {
    target.overrideEnabled = enabled;
    target.coolingActive = enabled;
    (target as any).manualDisengaged = !enabled;
    if (enabled) {
      // Preserve starting telemetry values without any instant jump.
      // Reset coolingProgress to 0 to begin the gradual reduction transition.
      (target as any).coolingProgress = 0.0;
    } else {
      (target as any).coolingProgress = 0.0;
      target.riskScore = Math.min(0.40, target.riskScore);
    }
  }
}

if (typeof window !== 'undefined') {
  (window as any).__setRackCooling = setRackCooling;
  (window as any).__racks = racks;
}
