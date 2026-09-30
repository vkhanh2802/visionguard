import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { api } from "./api";
import type {
  AnalysisRequest,
  EventRecord,
  Run,
  RunAnalytics,
} from "./types";

const defaultAnalysisRequest: AnalysisRequest = {
  source_path: "C:\\VisionGuard\\videos\\test.mp4",
  output_path: "C:\\VisionGuard\\outputs\\dashboard-output.mp4",
  config_path: "C:\\VisionGuard\\configs\\default.yaml",
};

function formatDate(value: string | null): string {
  if (!value) {
    return "Not finished";
  }

  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(value));
}

function formatNumber(value: number | null): string {
  return value === null ? "-" : value.toLocaleString();
}

function formatSeconds(value: number | null): string {
  return value === null ? "-" : `${value.toFixed(1)} s`;
}

function formatDecimal(value: number | null): string {
  return value === null ? "-" : value.toFixed(1);
}

function shortRunId(runId: string): string {
  return runId.slice(0, 8);
}

export function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<RunAnalytics | null>(null);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [health, setHealth] = useState<"checking" | "connected" | "offline">("checking");
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [analysisRequest, setAnalysisRequest] = useState(defaultAnalysisRequest);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? null;

  async function refreshRuns() {
    setIsRefreshing(true);
    try {
      const [healthResponse, runResponse] = await Promise.all([
        api.getHealth(),
        api.listRuns(),
      ]);
      setHealth(healthResponse.database === "connected" ? "connected" : "offline");
      setRuns(runResponse.items);
      setSelectedRunId((currentRunId) => {
        const currentRunExists = runResponse.items.some(
          (run) => run.run_id === currentRunId,
        );
        return currentRunExists ? currentRunId : runResponse.items[0]?.run_id ?? null;
      });
      setError(null);
    } catch (requestError) {
      setHealth("offline");
      setError(requestError instanceof Error ? requestError.message : "Unable to load runs.");
    } finally {
      setIsRefreshing(false);
    }
  }

  useEffect(() => {
    void refreshRuns();
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      setAnalytics(null);
      setEvents([]);
      return;
    }

    let isCurrent = true;
    Promise.all([api.getAnalytics(selectedRunId), api.listEvents(selectedRunId)])
      .then(([analyticsResponse, eventResponse]) => {
        if (isCurrent) {
          setAnalytics(analyticsResponse);
          setEvents(eventResponse.items);
        }
      })
      .catch((requestError) => {
        if (isCurrent) {
          setError(requestError instanceof Error ? requestError.message : "Unable to load run details.");
        }
      });

    return () => {
      isCurrent = false;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRun || !["queued", "running"].includes(selectedRun.status)) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshRuns();
    }, 5000);

    return () => window.clearInterval(intervalId);
  }, [selectedRun]);

  async function submitAnalysis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);

    try {
      const response = await api.startAnalysis(analysisRequest);
      setSelectedRunId(response.run_id);
      setError(null);
      await refreshRuns();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to start analysis.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">VIDEO ANALYTICS CONTROL ROOM</p>
          <h1>VisionGuard</h1>
        </div>
        <div className={`connection ${health}`}>
          <span className="connection-dot" />
          {health === "connected" ? "SQLite connected" : health === "offline" ? "API offline" : "Checking API"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="analysis-panel">
        <div className="analysis-panel-copy">
          <p className="eyebrow">NEW ANALYSIS</p>
          <h2>Turn a local video into a reviewable run.</h2>
          <p>Runs are persisted in SQLite. Processing status updates automatically while this dashboard is open.</p>
        </div>
        <form className="analysis-form" onSubmit={submitAnalysis}>
          <label>
            Source video
            <input
              value={analysisRequest.source_path}
              onChange={(event) => setAnalysisRequest({ ...analysisRequest, source_path: event.target.value })}
              required
            />
          </label>
          <label>
            Annotated output
            <input
              value={analysisRequest.output_path}
              onChange={(event) => setAnalysisRequest({ ...analysisRequest, output_path: event.target.value })}
              required
            />
          </label>
          <label>
            Configuration
            <input
              value={analysisRequest.config_path}
              onChange={(event) => setAnalysisRequest({ ...analysisRequest, config_path: event.target.value })}
              required
            />
          </label>
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Queueing run..." : "Start analysis"}
          </button>
        </form>
      </section>

      <section className="workspace">
        <aside className="run-list-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">RECENT ACTIVITY</p>
              <h2>Analysis runs</h2>
            </div>
            <button className="text-button" onClick={() => void refreshRuns()} disabled={isRefreshing}>
              {isRefreshing ? "Refreshing" : "Refresh"}
            </button>
          </div>

          <div className="run-list">
            {runs.length === 0 && <p className="empty-state">No persisted runs yet.</p>}
            {runs.map((run) => (
              <button
                className={`run-card ${run.run_id === selectedRunId ? "selected" : ""}`}
                key={run.run_id}
                onClick={() => setSelectedRunId(run.run_id)}
              >
                <span className={`status-dot ${run.status}`} />
                <span className="run-card-content">
                  <strong>{shortRunId(run.run_id)}</strong>
                  <span>{run.source_path}</span>
                  <small>{formatDate(run.created_at)}</small>
                </span>
                <span className={`status-label ${run.status}`}>{run.status}</span>
              </button>
            ))}
          </div>
        </aside>

        <section className="detail-panel">
          {!selectedRun && (
            <div className="empty-detail">
              <p className="eyebrow">NO RUN SELECTED</p>
              <h2>Choose a run to inspect its event timeline.</h2>
            </div>
          )}

          {selectedRun && (
            <>
              <div className="detail-header">
                <div>
                  <p className="eyebrow">RUN {shortRunId(selectedRun.run_id)}</p>
                  <h2>{selectedRun.source_path}</h2>
                  <p className="muted">Started {formatDate(selectedRun.created_at)}</p>
                </div>
                <div className="detail-actions">
                  <span className={`status-label ${selectedRun.status}`}>{selectedRun.status}</span>
                  {selectedRun.status === "completed" && (
                    <a className="download-button" href={api.outputUrl(selectedRun.run_id)}>
                      Download video
                    </a>
                  )}
                </div>
              </div>

              {selectedRun.error_message && <p className="run-error">{selectedRun.error_message}</p>}

              <div className="metric-grid">
                <Metric label="Frames" value={formatNumber(analytics?.processed_frames ?? selectedRun.processed_frames)} />
                <Metric label="In / out" value={`${formatNumber(analytics?.in_count ?? selectedRun.in_count)} / ${formatNumber(analytics?.out_count ?? selectedRun.out_count)}`} />
                <Metric label="Intrusions" value={formatNumber(analytics?.intrusion_count ?? selectedRun.intrusion_count)} />
                <Metric label="Loitering" value={formatNumber(analytics?.loitering_count ?? selectedRun.loitering_count)} />
                <Metric label="Events" value={formatNumber(analytics?.events.recorded_event_count ?? null)} />
                <Metric label="Elapsed" value={formatSeconds(selectedRun.elapsed_seconds)} />
              </div>

              <div className="detail-columns">
                <section className="timeline-panel">
                  <div className="section-heading">
                    <div>
                      <p className="eyebrow">EVENT LOG</p>
                      <h3>Observed activity</h3>
                    </div>
                    <span className="muted">{events.length} records</span>
                  </div>
                  <div className="event-table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Time</th>
                          <th>Event</th>
                          <th>Track</th>
                          <th>Context</th>
                        </tr>
                      </thead>
                      <tbody>
                        {events.map((event) => <EventRow event={event} key={event.event_id} />)}
                        {events.length === 0 && (
                          <tr>
                            <td colSpan={4} className="empty-cell">No events recorded for this run.</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="breakdown-panel">
                  <p className="eyebrow">EVENT BREAKDOWN</p>
                  <h3>Signals by type</h3>
                  <div className="event-breakdown">
                    {Object.entries(analytics?.events.by_type ?? {}).map(([eventType, count]) => (
                      <div className="breakdown-row" key={eventType}>
                        <span>{eventType.replace("_", " ")}</span>
                        <strong>{count}</strong>
                      </div>
                    ))}
                    {!analytics && <p className="muted">Loading event analytics...</p>}
                    {analytics && Object.keys(analytics.events.by_type).length === 0 && (
                      <p className="muted">No event types recorded.</p>
                    )}
                  </div>
                  <dl className="run-facts">
                    <div>
                      <dt>Unique tracks</dt>
                      <dd>{formatNumber(analytics?.events.unique_track_count ?? null)}</dd>
                    </div>
                    <div>
                      <dt>First event</dt>
                      <dd>{analytics?.events.first_event_timestamp?.toFixed(2) ?? "-"} s</dd>
                    </div>
                    <div>
                      <dt>Last event</dt>
                      <dd>{analytics?.events.last_event_timestamp?.toFixed(2) ?? "-"} s</dd>
                    </div>
                    <div>
                      <dt>Core FPS</dt>
                      <dd>{selectedRun.core_processing_fps?.toFixed(1) ?? "-"}</dd>
                    </div>
                  </dl>
                </section>
              </div>

              <section className="tracking-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">TRACKING DIAGNOSTICS</p>
                    <h3>ID continuity and detection gaps</h3>
                  </div>
                </div>
                {!analytics && <p className="muted">Loading tracking diagnostics...</p>}
                {analytics && !analytics.tracking && (
                  <p className="muted">Tracking diagnostics are available for newly completed runs.</p>
                )}
                {analytics?.tracking && (
                  <>
                    <div className="diagnostic-grid">
                      <Metric label="Total IDs" value={formatNumber(analytics.tracking.total_track_count)} />
                      <Metric label="New IDs in ROI" value={formatNumber(analytics.tracking.new_track_count_in_roi)} />
                      <Metric label="Tracks with gaps" value={formatNumber(analytics.tracking.tracks_with_gaps)} />
                      <Metric label="Longest gap" value={`${analytics.tracking.max_gap_frames} frames`} />
                      <Metric label="Missing frames" value={formatNumber(analytics.tracking.total_missing_frames)} />
                      <Metric label="Median observed" value={formatDecimal(analytics.tracking.median_observed_frames)} />
                    </div>
                    <div className="event-table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Track</th>
                            <th>Observed</th>
                            <th>Lifetime</th>
                            <th>Missing</th>
                            <th>Longest gap</th>
                            <th>Born in ROI</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...analytics.tracking.track_lifetimes]
                            .sort((left, right) => right.longest_gap_frames - left.longest_gap_frames || right.total_missing_frames - left.total_missing_frames)
                            .slice(0, 10)
                            .map((track) => (
                              <tr key={track.track_id}>
                                <td>#{track.track_id}</td>
                                <td>{track.observed_frames}</td>
                                <td>{track.lifetime_frames}</td>
                                <td>{track.total_missing_frames}</td>
                                <td>{track.longest_gap_frames}</td>
                                <td>{track.first_seen_in_roi ? "Yes" : "No"}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                    {analytics.tracking.continuity && (
                      <div className="continuity-summary">
                        <p className="eyebrow">CONTINUITY DECISIONS</p>
                        <div className="diagnostic-grid continuity-grid">
                          <Metric label="Reassociated" value={formatNumber(analytics.tracking.continuity.replacement_match_count)} />
                          <Metric label="Pending" value={formatNumber(analytics.tracking.continuity.pending_match_count)} />
                          <Metric label="Ambiguous rejected" value={formatNumber(analytics.tracking.continuity.rejected_ambiguous_match_count)} />
                          <Metric label="Gap rejected" value={formatNumber(analytics.tracking.continuity.rejected_gap_match_count)} />
                        </div>
                        <div className="event-table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Frame</th>
                                <th>Canonical</th>
                                <th>Replacement</th>
                                <th>Gap</th>
                                <th>Distance</th>
                              </tr>
                            </thead>
                            <tbody>
                              {analytics.tracking.continuity.replacement_matches.slice(-10).map((match) => (
                                <tr key={`${match.frame_id}-${match.replacement_track_id}`}>
                                  <td>{match.frame_id}</td>
                                  <td>#{match.canonical_track_id}</td>
                                  <td>#{match.replacement_track_id}</td>
                                  <td>{match.gap_frames}</td>
                                  <td>{match.distance_px.toFixed(1)} px</td>
                                </tr>
                              ))}
                              {analytics.tracking.continuity.replacement_matches.length === 0 && (
                                <tr>
                                  <td colSpan={5} className="empty-cell">No replacement IDs reassociated.</td>
                                </tr>
                              )}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    {analytics.tracking.deduplication && (
                      <div className="continuity-summary">
                        <p className="eyebrow">DUPLICATE SUPPRESSION</p>
                        <div className="diagnostic-grid continuity-grid">
                          <Metric label="Confirmed pairs" value={formatNumber(analytics.tracking.deduplication.confirmed_pair_count)} />
                          <Metric label="Suppressed boxes" value={formatNumber(analytics.tracking.deduplication.suppressed_observation_count)} />
                          <Metric label="Pending boxes" value={formatNumber(analytics.tracking.deduplication.pending_duplicate_observation_count)} />
                          <Metric label="Ambiguous pairs" value={formatNumber(analytics.tracking.deduplication.ambiguous_pair_count)} />
                        </div>
                        <div className="event-table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Confirmed frame</th>
                                <th>Primary</th>
                                <th>Duplicate</th>
                                <th>Containment</th>
                                <th>IoU</th>
                              </tr>
                            </thead>
                            <tbody>
                              {analytics.tracking.deduplication.confirmed_pairs.slice(-10).map((pair) => (
                                <tr key={`${pair.confirmed_frame_id}-${pair.duplicate_track_id}`}>
                                  <td>{pair.confirmed_frame_id}</td>
                                  <td>#{pair.primary_track_id}</td>
                                  <td>#{pair.duplicate_track_id}</td>
                                  <td>{(pair.containment_ratio * 100).toFixed(0)}%</td>
                                  <td>{pair.iou.toFixed(2)}</td>
                                </tr>
                              ))}
                              {analytics.tracking.deduplication.confirmed_pairs.length === 0 && (
                                <tr>
                                  <td colSpan={5} className="empty-cell">No active duplicate pairs confirmed.</td>
                                </tr>
                              )}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </section>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EventRow({ event }: { event: EventRecord }) {
  const context = [event.zone_id, event.direction, event.duration_seconds ? `${event.duration_seconds.toFixed(1)} s` : null]
    .filter(Boolean)
    .join(" / ");

  return (
    <tr>
      <td>{event.video_timestamp.toFixed(2)} s</td>
      <td><span className="event-tag">{event.event_type.replace("_", " ")}</span></td>
      <td>#{event.track_id}</td>
      <td>{context || "-"}</td>
    </tr>
  );
}
