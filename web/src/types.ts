export type RunStatus = "queued" | "running" | "completed" | "failed";

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
  tracking: {
    total_track_count: number;
    new_track_count_in_roi: number;
    tracks_with_gaps: number;
    total_missing_frames: number;
    max_gap_frames: number;
    median_observed_frames: number | null;
    track_lifetimes: TrackLifetime[];
    timing: TimingDiagnostics | null;
    continuity: ContinuityDiagnostics | null;
    deduplication: DeduplicationDiagnostics | null;
  } | null;
}

export interface TimingDiagnostics {
  read_seconds: number;
  tracking_seconds: number;
  analytics_seconds: number;
  drawing_seconds: number;
  write_enqueue_seconds: number;
  encoding_seconds: number;
  writer_flush_seconds: number;
  frame_loop_seconds: number;
  tracking_fps: number;
}

export interface TrackLifetime {
  track_id: number;
  first_seen_frame: number;
  last_seen_frame: number;
  observed_frames: number;
  lifetime_frames: number;
  total_missing_frames: number;
  longest_gap_frames: number;
  first_seen_in_roi: boolean;
}

export interface ContinuityDiagnostics {
  replacement_match_count: number;
  pending_match_count: number;
  rejected_ambiguous_match_count: number;
  rejected_gap_match_count: number;
  replacement_matches: ReplacementMatch[];
}

export interface ReplacementMatch {
  canonical_track_id: number;
  replacement_track_id: number;
  frame_id: number;
  gap_frames: number;
  distance_px: number;
}

export interface DeduplicationDiagnostics {
  confirmed_pair_count: number;
  pending_duplicate_observation_count: number;
  suppressed_observation_count: number;
  ambiguous_pair_count: number;
  handoff_count: number;
  confirmed_pairs: ConfirmedDuplicatePair[];
}

export interface ConfirmedDuplicatePair {
  primary_track_id: number;
  duplicate_track_id: number;
  confirmed_frame_id: number;
  containment_ratio: number;
  iou: number;
  bottom_distance_px: number;
}

export interface AnalysisRequest {
  source_path: string;
  config_path: string;
}

export interface AnalysisAcceptedResponse {
  run_id: string;
  status: "queued";
}
