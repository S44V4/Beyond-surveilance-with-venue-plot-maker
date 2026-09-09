export type Zone = {
  id: string;
  name: string;
  polygon: number[][];
  image_polygon: number[][] | null;
  capacity: number;
  area_m2: number | null;
  observed: boolean;
};
export type Portal = {
  id: string;
  name: string;
  source: string;
  target: string;
  kind: "entrance" | "exit" | "passage";
  position: number[];
  capacity: number;
  open: boolean;
  bidirectional: boolean;
  travel_seconds: number;
};
export type Venue = {
  id: string;
  name: string;
  version: number;
  description: string;
  geometry_kind: "schematic" | "reviewed_image_zones" | "calibrated";
  calibration_note: string;
  background: string | null;
  zones: Zone[];
  portals: Portal[];
};
export type ZoneState = {
  id: string;
  count: number | null;
  density: number | null;
  velocity: number | null;
  direction: number[] | null;
  acceleration: number | null;
  occupancy_ratio: number | null;
  pressure: number | null;
};
export type Forecast = {
  horizon: number;
  unit: string;
  count: number;
  zones: { id: string; count: number | null; pressure?: number }[];
};
export type Snapshot = {
  index: number;
  timestamp: number;
  count: number;
  mapped_count: number;
  unmapped_count: number;
  zones: ZoneState[];
  heatmap: number[][];
  points: number[][];
  frame_url?: string;
  forecasts: Forecast[];
  forecast_method?: string;
  learned: {
    forecasts: Forecast[];
    provenance: Record<string, unknown>;
  } | null;
  learned_status: string;
  quality: string[];
  evidence: string;
  latency_ms: number;
  units: Record<string, string>;
};
export type Video = {
  id: string;
  name: string;
  poster: string;
  url: string;
  size: number;
  fps: number;
  duration: number;
  frame_count: number;
  width: number;
  height: number;
};
export type Job = {
  id: string;
  kind: string;
  status: string;
  progress: number;
  error: string | null;
  created: number;
  updated: number;
  body: {
    venue: Venue;
    venue_id?: string;
    video_id?: string;
    sample_fps?: number;
    camera_name?: string;
    example?: boolean;
    request?: ScenarioInput;
  };
  video?: Video | null;
  result?: Record<string, unknown> | null;
};
export type GateChanges = Record<string, { open: boolean; capacity?: number }>;
export type ScenarioInput = {
  run_id: string;
  snapshot_index: number;
  name: string;
  gates: GateChanges;
  duration: number;
  start: number;
  end: number | null;
  mode: "evacuation" | "continuous";
  inflow_per_second: number;
  unobserved_counts: Record<string, number>;
  sensitivity: boolean;
  idempotency_key: string;
};
export type Trajectory = {
  time: number;
  counts: Record<string, number>;
  total: number;
  exited: number;
  outside_queue: number;
  unreachable: number;
  gate_open: Record<string, boolean>;
  gate_totals: Record<string, number>;
  conservation_residual: number;
};
export type Metrics = {
  initial: number;
  remaining: number;
  departed: number;
  arrivals: number;
  outside_queue: number;
  peak_queue: number;
  peak_occupancy: number;
  peak_zone_count: number;
  clearance_seconds: number | null;
  unreachable: number;
  conservation_error: number;
  gate_totals: Record<string, number>;
};
export type ScenarioResult = {
  id: string;
  name: string;
  run_id: string;
  source_timestamp: number;
  source_evidence: string;
  venue: Venue;
  request: ScenarioInput;
  engine: string;
  evidence: string;
  baseline: { trajectory: Trajectory[]; metrics: Metrics };
  scenario: { trajectory: Trajectory[]; metrics: Metrics };
  sensitivity: {
    capacity_factor: number;
    baseline_remaining: number;
    scenario_remaining: number;
    remaining_delta: number;
  }[];
  delta: Record<string, number>;
  assumptions: string[];
};
export type Health = {
  status: string;
  worker_alive: boolean;
  version: string;
  checkpoints_present: Record<string, boolean>;
  models: {
    ready: boolean;
    device: string;
    ddpf: {
      kind: string;
      sha256: string;
      datasets: string[];
      image_size: number[];
    };
    temporal: {
      ready: boolean;
      error: string | null;
      venue_id: string;
      unit: string;
      sha256: string;
      graph_sha256: string;
    };
    verified_at: number;
  } | null;
};
