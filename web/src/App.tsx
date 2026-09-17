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
    if (!selectedRun || selectedRun.status !== "running") {
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
