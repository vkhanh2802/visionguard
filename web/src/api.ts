import type {
  AnalysisAcceptedResponse,
  AnalysisRequest,
  EventListResponse,
  HealthResponse,
  RunAnalytics,
  RunListResponse,
} from "./types";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, options);

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail ?? `Request failed with status ${response.status}.`;
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  getHealth: () => request<HealthResponse>("/health"),
  listRuns: () => request<RunListResponse>("/runs?limit=100"),
  getAnalytics: (runId: string) => request<RunAnalytics>(`/runs/${runId}/analytics`),
  listEvents: (runId: string) =>
    request<EventListResponse>(`/events?run_id=${encodeURIComponent(runId)}&limit=100`),
  startAnalysis: (payload: AnalysisRequest) =>
    request<AnalysisAcceptedResponse>("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  outputUrl: (runId: string) => `${apiBaseUrl}/runs/${runId}/output`,
};
