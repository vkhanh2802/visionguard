export type RunStatus = "running" | "completed" | "failed";

export interface HealthResponse {
  status: "ok";
  database: "connected";
}

export interface Run {
  run_id: string;
  source_path: string;
  output_path: string;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
  error_message: string | null;
  processed_frames: number | null;
  source_fps: number | null;
  effective_fps: number | null;
  core_processing_fps: number | null;
  end_to_end_fps: number | null;
  elapsed_seconds: number | null;
  stopped_early: boolean | null;
  in_count: number | null;
  out_count: number | null;
  intrusion_count: number | null;
  loitering_count: number | null;
}

export interface RunListResponse {
  items: Run[];
  limit: number;
  offset: number;
}

export interface EventRecord {
  event_id: number;
  run_id: string;
  frame_id: number;
  event_type: string;
  track_id: number;
  video_timestamp: number;
  position_x: number;
  position_y: number;
  direction: string | null;
  zone_id: string | null;
  duration_seconds: number | null;
  logged_at: string;
}

export interface EventListResponse {
  items: EventRecord[];
  limit: number;
  offset: number;
}

export interface RunAnalytics {
  run_id: string;
  status: RunStatus;
  processed_frames: number | null;
  in_count: number | null;
  out_count: number | null;
  net_count: number | null;
  intrusion_count: number | null;
  loitering_count: number | null;
  events: {
    recorded_event_count: number;
    unique_track_count: number;
    first_event_timestamp: number | null;
    last_event_timestamp: number | null;
    by_type: Record<string, number>;
  };
}

export interface AnalysisRequest {
  source_path: string;
  output_path: string;
  config_path: string;
}

export interface AnalysisAcceptedResponse {
  run_id: string;
  status: "running";
}
