// GET /telemetry
export interface ThervoTelemetry {
  operating_mode: 'LOCAL_LAPTOP' | 'DATA_CENTER_SIMULATION';
  racks: Array<{
    id: string; // e.g., "A07"
    telemetry: {
      cpu_util: number;
      gpu_util: number;
      cpu_temp: number;
    };
    risk_score: number;
    cooling: {
      target_rpm: number;
      actual_rpm: number;
      status: string;
    };
    ai_insights?: {
      gnn_embed: number;
      xgb_pred: number;
      zone: string;
    };
  }>;
  topology?: Array<{
    source: string;
    target: string;
    weight: number;
  }>; // GNN edges
  global_health: {
    status: string;
    issues: string[];
  };
  events: Array<{
    time: string;
    message: string;
  }>;
}

// POST /simulation/controls
export interface ScenarioParams {
  simulated_load: number;
  ambient_temp_offset: number;
  trigger_spike: boolean;
  mode?: 'LOCAL_LAPTOP' | 'DATA_CENTER_SIMULATION';
}
