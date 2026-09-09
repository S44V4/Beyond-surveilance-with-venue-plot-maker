import { useCallback, useEffect, useRef, useState } from "react";
import { api, clock, idempotency, number, post } from "./api";
import type {
  GateChanges,
  Health,
  Job,
  ScenarioInput,
  ScenarioResult,
  Snapshot,
  Venue,
} from "./types";
import { Button, Empty, Field, Icon, Panel, Select, Status } from "./ui";
import { VenueMap } from "./Map";
import { Chart } from "./Chart";
import { UploadDialog } from "./Upload";
import { ScenarioControls } from "./ScenarioControls";
import { ScenarioView } from "./ScenarioView";
import { VenueEditor } from "./VenueEditor";

const views = [
  { id: "overview", name: "Overview", icon: "grid" },
  { id: "scenarios", name: "Scenario lab", icon: "branch" },
  { id: "venue", name: "Venue setup", icon: "map" },
  { id: "library", name: "Run library", icon: "video" },
  { id: "models", name: "Model evidence", icon: "layers" },
];
const active = (job: Job | null) =>
  !!job && ["queued", "running", "cancelling"].includes(job.status);

function Heatmap({ values }: { values: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!values?.length || !ref.current) return;
    const canvas = ref.current;
    canvas.width = values[0].length;
    canvas.height = values.length;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const pixels = ctx.createImageData(canvas.width, canvas.height);
    for (let y = 0; y < canvas.height; y++)
      for (let x = 0; x < canvas.width; x++) {
        const v = values[y][x],
          i = (y * canvas.width + x) * 4;
        pixels.data[i] = Math.round(230 * v);
        pixels.data[i + 1] = Math.round(150 + 60 * (1 - v));
        pixels.data[i + 2] = Math.round(245 * (1 - v));
        pixels.data[i + 3] = v > 0.05 ? Math.round(v * 170) : 0;
      }
    ctx.putImageData(pixels, 0, 0);
  }, [values]);
  return <canvas ref={ref} className="heatmap-overlay" aria-hidden="true" />;
}

function VideoPanel({
  run,
  snapshot,
  timeline,
  onSelect,
  onUpload,
}: {
  run: Job | null;
  snapshot: Snapshot | null;
  timeline: Snapshot[];
  onSelect: (n: number) => void;
  onUpload: () => void;
}) {
  const [layer, setLayer] = useState("density"),
    [original, setOriginal] = useState(false),
    [failed, setFailed] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    setOriginal(false);
    setFailed(false);
  }, [run?.id]);
  const frame = snapshot?.frame_url || run?.video?.poster;
  return (
    <Panel
      title="Camera view"
      action={
        run?.video ? (
          <span className="badge">
            <Icon name="video" size={13} />
            {run.body.camera_name}
          </span>
        ) : undefined
      }
      className="video-panel"
    >
      {run?.video ? (
        <>
          <div
            className="video-frame"
            style={{ aspectRatio: `${run.video.width}/${run.video.height}` }}
          >
            {original && !failed ? (
              <video
                ref={video}
                src={run.video.url}
                poster={run.video.poster}
                controls
                onError={() => {
                  setFailed(true);
                  setOriginal(false);
                }}
                onTimeUpdate={() => {
                  const t = video.current?.currentTime || 0;
                  let nearest = 0;
                  for (let i = 1; i < timeline.length; i++) {
                    if (
                      Math.abs(timeline[i].timestamp - t) <
                      Math.abs(timeline[nearest].timestamp - t)
                    )
                      nearest = i;
                  }
                  if (timeline[nearest]) onSelect(timeline[nearest].index);
                }}
              />
            ) : (
              <>
                {frame && (
                  <img
                    src={frame}
                    alt={`Analyzed frame at ${snapshot ? clock(snapshot.timestamp) : "start"}`}
                  />
                )}
                <div className="video-corner">
                  <span className="record-square" />
                  {snapshot ? clock(snapshot.timestamp) : "00:00"} · analysis
                  frame
                </div>
                {layer === "density" && snapshot?.heatmap?.length ? (
                  <Heatmap values={snapshot.heatmap} />
                ) : null}
                {layer === "points" && snapshot && (
                  <svg
                    className="point-overlay"
                    viewBox="0 0 1000 1000"
                    preserveAspectRatio="none"
                    aria-label="Estimated head locations"
                  >
                    {snapshot.points?.map(([x, y], i) => (
                      <circle
                        key={i}
                        cx={x}
                        cy={y}
                        r="5"
                        fill="none"
                        stroke="#57f3cf"
                        strokeWidth="2"
                      />
                    ))}
                  </svg>
                )}
              </>
            )}
          </div>
          <div className="video-tools">
            <div className="segmented">
              {["density", "points", "none"].map((l) => (
                <button
                  key={l}
                  aria-pressed={layer === l && !original}
                  onClick={() => {
                    setLayer(l);
                    setOriginal(false);
                  }}
                >
                  {l === "none"
                    ? "Clean"
                    : l === "points"
                      ? "Heads"
                      : "Density"}
                </button>
              ))}
            </div>
            <button
              className="text-button"
              onClick={() => setOriginal(!original)}
            >
              {original ? "Analysis frames" : "Play original"}
            </button>
          </div>
          {failed && (
            <p className="panel-note">
              This codec cannot play in your browser. Analysis frames remain
              available.
            </p>
          )}
          <div className="video-meta">
            <span>
              {run.video.width} × {run.video.height}
            </span>
            <span>{number(run.video.fps, 1)} FPS</span>
            <span>{number(run.body.sample_fps, 1)} analysis FPS</span>
          </div>
        </>
      ) : (
        <Empty
          icon="video"
          title={
            run?.body.example
              ? "Illustrative starting state"
              : "Your footage. A clearer picture."
          }
          description={
            run?.body.example
              ? "This study uses assumed counts to demonstrate gate-flow comparisons. Upload a video to use model estimates."
              : "Upload a video to inspect density, head locations and crowd movement over time."
          }
          action={
            <Button onClick={onUpload}>
              <Icon name="upload" size={16} />
              Upload video
            </Button>
          }
        />
      )}
    </Panel>
  );
}

export default function App() {
  const [view, setView] = useState(
    () => new URLSearchParams(location.search).get("view") || "overview",
  );
  const [health, setHealth] = useState<Health | null>(null),
    [venues, setVenues] = useState<Venue[]>([]),
    [venueId, setVenueId] = useState(""),
    [runs, setRuns] = useState<Job[]>([]),
    [run, setRun] = useState<Job | null>(null),
    [timeline, setTimeline] = useState<Snapshot[]>([]),
    [snapshot, setSnapshot] = useState<Snapshot | null>(null),
    [selectedIndex, setSelectedIndex] = useState(0),
    [zone, setZone] = useState("Z05"),
    [changes, setChanges] = useState<GateChanges>({}),
    [uploadOpen, setUploadOpen] = useState(false),
    [notification, setNotification] = useState<{
      message: string;
      tone: string;
    } | null>(null),
    [offline, setOffline] = useState(false),
    [search, setSearch] = useState(""),
    [heat, setHeat] = useState(true),
    [connections, setConnections] = useState(true),
    [scenarioId, setScenarioId] = useState(""),
    [scenarioJob, setScenarioJob] = useState<Job | null>(null),
    [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null),
    [history, setHistory] = useState<Job[]>([]),
    [scenarioBusy, setScenarioBusy] = useState(false),
    [libraryLimit, setLibraryLimit] = useState(30),
    [refresh, setRefresh] = useState(0);
  const framesRef = useRef<Snapshot[]>([]),
    runKey = run?.id;
  const venue =
    run?.body.venue || venues.find((v) => v.id === venueId) || venues[0];
  const navigate = useCallback((v: string) => {
    setView(v);
    historyReplace(v);
    window.scrollTo({ top: 0, behavior: "instant" });
  }, []);
  function notify(message: string, tone = "info") {
    setNotification({ message, tone });
  }
  function openRun(j: Job) {
    setNotification(null);
    setRefresh((x) => x + 1);
    setRun(j);
    setTimeline([]);
    framesRef.current = [];
    setSnapshot(null);
    setSelectedIndex(0);
    setChanges({});
    setVenueId(j.body.venue.id);
    localStorage.setItem("bs-run", j.id);
    navigate("overview");
  }
  useEffect(() => {
    const pop = () =>
      setView(new URLSearchParams(location.search).get("view") || "overview");
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  useEffect(() => {
    let stopped = false;
    async function load() {
      try {
        const [h, v, r, s] = await Promise.all([
          api<Health>("/health"),
          api<Venue[]>("/venues"),
          api<Job[]>(`/runs?limit=${libraryLimit}`),
          api<Job[]>("/scenarios"),
        ]);
        if (stopped) return;
        setHealth(h);
        setVenues(v);
        setRuns(r);
        setHistory(s);
        setOffline(false);
        if (!runKey) {
          const saved = localStorage.getItem("bs-run");
          const found = r.find((x) => x.id === saved);
          if (found) {
            setRun(found);
            setVenueId(found.body.venue.id);
          }
        }
      } catch {
        if (!stopped) setOffline(true);
      }
    }
    load();
    const timer = setInterval(load, 10000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [refresh, libraryLimit, runKey]);
  useEffect(() => {
    document.title = `${views.find((v) => v.id === view)?.name || "Overview"} · Beyond Surveillance`;
  }, [view]);
  useEffect(() => {
    if (!runKey) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const [j, data] = await Promise.all([
          api<Job>(`/runs/${runKey}`),
          api<{ items: Snapshot[]; total: number }>(
            `/runs/${runKey}/timeline?offset=${framesRef.current.length}&limit=1000`,
          ),
        ]);
        if (stopped) return;
        setRun(j);
        if (data.items.length) {
          framesRef.current = [...framesRef.current, ...data.items];
          setTimeline(framesRef.current);
        }
        if (active(j) || framesRef.current.length < data.total)
          timer = setTimeout(
            poll,
            framesRef.current.length < data.total ? 100 : 2000,
          );
      } catch (e) {
        if (!stopped) {
          notify((e as Error).message, "error");
          timer = setTimeout(poll, 6000);
        }
      }
    }
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [runKey, refresh]);
  useEffect(() => {
    if (!runKey || !timeline.length) return;
    const controller = new AbortController();
    api<Snapshot>(`/runs/${runKey}/snapshots/${selectedIndex}`, {
      signal: controller.signal,
    })
      .then(setSnapshot)
      .catch((e) => {
        if (!controller.signal.aborted) notify(e.message, "error");
      });
    return () => controller.abort();
  }, [runKey, selectedIndex, timeline.length > 0]);
  useEffect(() => {
    if (!scenarioId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const { result, ...job } = await api<
          Omit<Job, "result"> & { result: ScenarioResult | null }
        >(`/scenarios/${scenarioId}`);
        if (stopped) return;
        setScenarioJob(job);
        setScenarioResult(result);
        if (active(job)) timer = setTimeout(poll, 800);
        else {
          setScenarioBusy(false);
          setRefresh((x) => x + 1);
          if (job.error) notify(job.error, "error");
        }
      } catch (e) {
        if (!stopped) {
          setScenarioBusy(false);
          notify((e as Error).message, "error");
        }
      }
    }
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [scenarioId]);
  async function example() {
    try {
      const j = await post<Job>("/examples");
      openRun(j);
      notify(
        "Illustrative state loaded. Counts and gate capacities are assumptions.",
      );
      setRefresh((x) => x + 1);
    } catch (e) {
      notify((e as Error).message, "error");
    }
  }
  async function compare(input: ScenarioInput) {
    setScenarioBusy(true);
    try {
      const j = await post<Job>("/scenarios", input);
      setScenarioJob(j);
      setScenarioResult(null);
      setScenarioId(j.id);
      navigate("scenarios");
    } catch (e) {
      setScenarioBusy(false);
      notify((e as Error).message, "error");
    }
  }
  async function cancel(id: string) {
    try {
      await post(`/jobs/${id}/cancel`);
      setRefresh((x) => x + 1);
      notify("Cancellation requested. Completed snapshots remain available.");
    } catch (e) {
      notify((e as Error).message, "error");
    }
  }
  async function retry(j: Job) {
    try {
      const next = await post<Job>(`/jobs/${j.id}/retry`);
      openRun(next);
      setRefresh((x) => x + 1);
    } catch (e) {
      notify((e as Error).message, "error");
    }
  }
  const z = venue?.zones.find((z) => z.id === zone) || venue?.zones[0],
    zs = snapshot?.zones.find((s) => s.id === z?.id);
  const gates = venue?.portals.filter((p) => p.kind !== "passage") || [];
  const openExits = gates.filter((p) => p.kind === "exit" && p.open).length;
  const forecasts = snapshot?.learned?.forecasts || snapshot?.forecasts || [];
  const filteredRuns = runs.filter((r) =>
    (r.video?.name || r.body.camera_name || "")
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  const metricCards = [
    {
      title:
        snapshot?.evidence === "hypothetical_scenario_input"
          ? "Assumed population"
          : "Estimated population",
      value: number(snapshot?.mapped_count),
      unit: "people in mapped zones",
      icon: "people",
      foot:
        snapshot?.evidence === "hypothetical_scenario_input"
          ? "Illustrative starting state"
          : snapshot
            ? "From your DDPF checkpoint"
            : "Awaiting video analysis",
    },
    {
      title: "Active exits",
      value: venue
        ? `${openExits} / ${gates.filter((p) => p.kind === "exit").length}`
        : "—",
      unit: "open in baseline",
      icon: "gate",
      foot:
        venue?.geometry_kind === "schematic"
          ? "Gate locations are assumptions"
          : "Reviewed venue configuration",
    },
    {
      title: "Observed zones",
      value: venue
        ? `${venue.zones.filter((z) => z.observed).length} / ${venue.zones.length}`
        : "—",
      unit: "camera coverage",
      icon: "map",
      foot: snapshot
        ? `${number(snapshot.unmapped_count, 1)} people outside mapped zones`
        : "Define coverage in venue setup",
    },
    {
      title: "Forecast method",
      value: snapshot?.learned
        ? "STRFE + graph"
        : snapshot?.forecasts?.length
          ? "Persistence"
          : "—",
      unit: snapshot?.learned ? "research model" : "continuation baseline",
      icon: "pulse",
      foot: snapshot?.learned
        ? "Frame-step horizons"
        : snapshot?.forecasts?.length
          ? "No assumed neural improvement"
          : "Available after video processing",
    },
  ];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          href="?view=overview"
          className="brand"
          onClick={(e) => {
            e.preventDefault();
            navigate("overview");
          }}
        >
          <span className="brand-mark">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            BEYOND<span>SURVEILLANCE</span>
          </span>
        </a>
        <div className="workspace-label">CROWD INTELLIGENCE</div>
        <nav aria-label="Main navigation">
          {views.map((v) => (
            <a
              key={v.id}
              href={`?view=${v.id}`}
              aria-current={view === v.id ? "page" : undefined}
              className={view === v.id ? "active" : ""}
              onClick={(e) => {
                e.preventDefault();
                navigate(v.id);
              }}
            >
              <Icon name={v.icon} size={19} />
              <span>{v.name}</span>
              {v.id === "scenarios" && history.length > 0 && (
                <small>{history.length}</small>
              )}
            </a>
          ))}
        </nav>
        <div className="sidebar-study">
          <Icon name="branch" size={24} />
          <strong>Explore the possibilities.</strong>
          <p>See how opening an exit changes crowd movement.</p>
          <button onClick={example}>
            Open example study
            <Icon name="arrow" size={16} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <span className="operator-avatar">L</span>
          <div>
            <strong>Local workspace</strong>
            <small>Research environment</small>
          </div>
          <Icon name="shield" size={17} />
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>{" "}
            <strong>
              {views.find((v) => v.id === view)?.name || "Overview"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className={`engine-status ${offline ? "offline" : ""}`}>
              <i />
              {offline
                ? "Disconnected"
                : health?.worker_alive
                  ? "Local engine connected"
                  : "Connecting to engine"}
            </span>
            <span className="version">v1.0</span>
            <span className="avatar">BS</span>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <span className="eyebrow">BEYOND THE FRAME</span>
              <h1>
                {view === "overview"
                  ? "Crowd overview"
                  : view === "scenarios"
                    ? "Scenario lab"
                    : view === "venue"
                      ? "Venue setup"
                      : view === "library"
                        ? "Run library"
                        : "Model evidence"}
              </h1>
              <p>
                {view === "overview"
                  ? "Understand the present. Explore what happens next."
                  : view === "scenarios"
                    ? "Same starting point. Different gate decisions."
                    : view === "venue"
                      ? "Connect camera observations to the places people move through."
                      : view === "library"
                        ? "Your footage, analysis history and saved comparisons."
                        : "Know what produced every prediction."}
              </p>
            </div>
            <div className="heading-actions">
              {run && (
                <a
                  className="button secondary"
                  href={`/api/runs/${run.id}/export`}
                  download
                >
                  <Icon name="download" size={17} />
                  Export run
                </a>
              )}
              <Button
                variant="primary"
                onClick={() => setUploadOpen(true)}
                disabled={!venues.length}
              >
                <Icon name="upload" size={17} />
                Analyze video
              </Button>
            </div>
          </div>
          {offline && (
            <Status
              tone="error"
              message="The local API is unavailable. Start the backend, then retry. Saved data will remain intact."
              closeLabel="Retry connection"
              onClose={() => setRefresh((x) => x + 1)}
            />
          )}
          {notification && (
            <Status {...notification} onClose={() => setNotification(null)} />
          )}
          {view === "overview" && (
            <>
              <div className="mobile-example">
                <Button onClick={example}>Open example study</Button>
              </div>
              <div className="context-bar">
                <div>
                  <Icon name="map" size={18} />
                  <label htmlFor="active-venue" className="sr-only">
                    Active venue
                  </label>
                  <Select
                    id="active-venue"
                    value={venue?.id || ""}
                    onChange={(e) => {
                      setVenueId(e.target.value);
                      setRun(null);
                      setSnapshot(null);
                      setTimeline([]);
                      framesRef.current = [];
                      localStorage.removeItem("bs-run");
                      setChanges({});
                    }}
                  >
                    {venues.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.name}
                      </option>
                    ))}
                  </Select>
                  <span className="badge">
                    {venue?.geometry_kind === "calibrated"
                      ? "Calibrated"
                      : venue?.geometry_kind === "reviewed_image_zones"
                        ? "Reviewed image zones"
                        : "Schematic map"}
                  </span>
                </div>
                <span className="context-source">
                  <Icon name="video" size={15} />
                  {run?.video?.name ||
                    (run?.body.example
                      ? "Illustrative scenario input"
                      : "No video selected")}
                </span>
              </div>
              <div className="metrics-grid">
                {metricCards.map((m) => (
                  <div className="metric-card" key={m.title}>
                    <div>
                      <span>{m.title}</span>
                      <Icon name={m.icon} size={19} />
                    </div>
                    <strong
                      className={m.value.length > 12 ? "small-value" : ""}
                    >
                      {m.value}
                    </strong>
                    <span className="metric-unit">{m.unit}</span>
                    <small>{m.foot}</small>
                  </div>
                ))}
              </div>
              {active(run) && (
                <div className="job-progress">
                  <span className="spinner" />
                  <div>
                    <strong>
                      {run?.status === "queued"
                        ? "Queued for processing"
                        : run?.status === "cancelling"
                          ? "Cancelling…"
                          : "Analyzing your video"}
                    </strong>
                    <span>
                      {timeline.length} snapshots ready ·{" "}
                      {Math.round((run?.progress || 0) * 100)}% processed
                    </span>
                    <div className="progress-track">
                      <i style={{ width: `${(run?.progress || 0) * 100}%` }} />
                    </div>
                  </div>
                  <Button
                    onClick={() => run && cancel(run.id)}
                    disabled={run?.status === "cancelling"}
                  >
                    Cancel
                  </Button>
                </div>
              )}
              {run?.status === "failed" && (
                <Status
                  tone="error"
                  message={
                    run.error ||
                    "Analysis failed. Open the run library to retry."
                  }
                />
              )}
              <div className="operations-grid">
                <div className="operations-main">
                  <Panel
                    title="Venue intelligence"
                    eyebrow="SPATIAL OVERVIEW"
                    action={
                      <div className="map-toggles">
                        <button
                          aria-pressed={heat}
                          onClick={() => setHeat(!heat)}
                        >
                          <Icon name="layers" size={15} />
                          Occupancy
                        </button>
                        <button
                          aria-pressed={connections}
                          onClick={() => setConnections(!connections)}
                        >
                          Connections
                        </button>
                      </div>
                    }
                    className="main-map-panel"
                  >
                    {venue ? (
                      <VenueMap
                        venue={venue}
                        states={snapshot?.zones}
                        selected={zone}
                        onSelect={setZone}
                        onGate={(id) => {
                          const p = venue.portals.find((p) => p.id === id)!;
                          setChanges({
                            ...changes,
                            [id]: {
                              ...changes[id],
                              open: !(changes[id]?.open ?? p.open),
                            },
                          });
                        }}
                        changes={changes}
                        heat={heat}
                        connections={connections}
                      />
                    ) : (
                      <Empty
                        icon="map"
                        title="Loading venue"
                        description="Connecting to the local workspace."
                      />
                    )}
                    <div className="map-footer">
                      <span>
                        <i className="legend-dot open" />
                        Open gate
                      </span>
                      <span>
                        <i className="legend-dot closed" />
                        Closed gate
                      </span>
                      <span>
                        <i className="legend-dot changed" />
                        Scenario change
                      </span>
                      <div className="density-scale">
                        <span>Low</span>
                        <i />
                        <span>High occupancy</span>
                      </div>
                    </div>
                  </Panel>
                  <Panel
                    title="Crowd timeline"
                    action={
                      <span className="badge">{timeline.length} snapshots</span>
                    }
                  >
                    <div className="timeline-body">
                      {timeline.length ? (
                        <Chart
                          series={[
                            {
                              name:
                                snapshot?.evidence ===
                                "hypothetical_scenario_input"
                                  ? "Assumed population"
                                  : "Estimated population",
                              values: timeline.map((f) => f.mapped_count),
                              color: "#2456d6",
                            },
                          ]}
                          labels={[
                            "00:00",
                            clock((timeline.at(-1)?.timestamp || 0) / 2),
                            clock(timeline.at(-1)?.timestamp || 0),
                          ]}
                        />
                      ) : (
                        <div className="timeline-empty">
                          <Icon name="pulse" size={26} />
                          <span>
                            Your crowd timeline will appear as video frames are
                            processed.
                          </span>
                        </div>
                      )}
                      <div className="playback">
                        <Icon name="clock" size={17} />
                        <span className="mono">
                          {clock(snapshot?.timestamp || 0)}
                        </span>
                        <input
                          aria-label="Video analysis timeline"
                          type="range"
                          min="0"
                          max={Math.max(0, timeline.length - 1)}
                          value={Math.min(
                            selectedIndex,
                            Math.max(0, timeline.length - 1),
                          )}
                          disabled={!timeline.length}
                          onChange={(e) =>
                            setSelectedIndex(Number(e.target.value))
                          }
                        />
                        <span className="mono">
                          {clock(timeline.at(-1)?.timestamp || 0)}
                        </span>
                      </div>
                    </div>
                  </Panel>
                </div>
                <div className="operations-inspector">
                  {venue && (
                    <Panel>
                      <ScenarioControls
                        key={runKey || venue.id}
                        venue={venue}
                        snapshot={snapshot}
                        run={run}
                        changes={changes}
                        setChanges={setChanges}
                        onRun={compare}
                        busy={scenarioBusy}
                      />
                    </Panel>
                  )}
                  <Panel
                    title="Zone focus"
                    action={<span className="badge blue">{z?.id || "—"}</span>}
                  >
                    <div className="panel-padding">
                      <Field label="Inspect a zone" id="focus-zone">
                        <Select
                          id="focus-zone"
                          value={z?.id || ""}
                          onChange={(e) => setZone(e.target.value)}
                        >
                          {venue?.zones.map((z) => (
                            <option key={z.id} value={z.id}>
                              {z.name}
                            </option>
                          ))}
                        </Select>
                      </Field>
                      <div className="zone-stats">
                        <div>
                          <span>Population</span>
                          <strong>
                            {number(zs?.count, 1)}
                            <small>people</small>
                          </strong>
                        </div>
                        <div>
                          <span>Image motion</span>
                          <strong>
                            {number(zs?.velocity, 1)}
                            <small>px/s</small>
                          </strong>
                        </div>
                        <div>
                          <span>Physical density</span>
                          <strong>
                            {number(zs?.density, 2)}
                            <small>people/m²</small>
                          </strong>
                        </div>
                        <div>
                          <span>Pressure proxy</span>
                          <strong>
                            {number(zs?.pressure, 1)}
                            <small>/ 100</small>
                          </strong>
                        </div>
                      </div>
                      <p className="micro-copy">
                        Physical density requires calibration. Pressure is an
                        ordinal model proxy, not an emergency probability.
                      </p>
                    </div>
                  </Panel>
                </div>
              </div>
              <div className="overview-bottom">
                <VideoPanel
                  run={run}
                  snapshot={snapshot}
                  timeline={timeline}
                  onSelect={setSelectedIndex}
                  onUpload={() => setUploadOpen(true)}
                />
                <Panel
                  title="Looking ahead"
                  eyebrow="OBSERVATIONAL FORECAST"
                  action={<Icon name="pulse" size={20} />}
                >
                  <div className="forecast-body">
                    {forecasts.length ? (
                      <>
                        <p>
                          {snapshot?.learned
                            ? "STRFE + graph research predictions. Horizons are normalized frame steps."
                            : "Persistence baseline: current population carried forward. This is an explicit baseline, not a trained temporal prediction."}
                        </p>
                        <div className="forecast-cards">
                          {forecasts.map((f) => (
                            <div key={f.horizon}>
                              <span>
                                +{f.horizon}{" "}
                                {f.unit === "seconds" ? "sec" : "steps"}
                              </span>
                              <strong>{number(f.count)}</strong>
                              <small>estimated people</small>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <Empty
                        icon="pulse"
                        title="Predictions need observations"
                        description="Process a video to create timestamped estimates and continuation forecasts."
                      />
                    )}
                    <div className="evidence-note">
                      <Icon name="info" size={17} />
                      <div>
                        <strong>Evidence stays attached</strong>
                        <p>
                          {snapshot?.learned_status ||
                            "Every result records its model, source timestamp, units and venue version."}
                        </p>
                      </div>
                    </div>
                    {snapshot?.quality.map((q) => (
                      <p className="quality-note" key={q}>
                        {q}
                      </p>
                    ))}
                  </div>
                </Panel>
              </div>
            </>
          )}
          {view === "scenarios" && (
            <>
              <ScenarioView
                job={scenarioJob}
                result={scenarioResult}
                onBack={() => navigate("overview")}
                onCancel={() => scenarioJob && cancel(scenarioJob.id)}
              />
              {history.length > 0 && (
                <Panel title="Saved comparisons">
                  <div className="history-list">
                    {history.map((j) => (
                      <button
                        key={j.id}
                        onClick={() => {
                          setScenarioResult(null);
                          setScenarioId(j.id);
                        }}
                      >
                        <span className="history-symbol">
                          <Icon name="branch" />
                        </span>
                        <div>
                          <strong>{j.body.request?.name || "Scenario"}</strong>
                          <small>
                            {new Date(j.created * 1000).toLocaleString("en-IN")}{" "}
                            · {j.body.request?.duration}s
                          </small>
                        </div>
                        <span
                          className={`badge ${j.status === "completed" ? "green" : ""}`}
                        >
                          {j.status}
                        </span>
                        <Icon name="arrow" size={16} />
                      </button>
                    ))}
                  </div>
                </Panel>
              )}
            </>
          )}
          {view === "venue" && (
            <VenueEditor
              venues={venues}
              onSave={(v) => {
                setVenues((old) => [v, ...old.filter((x) => x.id !== v.id)]);
                notify(
                  "Venue saved. Existing analysis runs retain their original mapping.",
                  "success",
                );
              }}
            />
          )}
          {view === "library" && (
            <Panel
              title="Analysis runs"
              action={<span className="badge">{runs.length} loaded</span>}
            >
              <div className="library-toolbar">
                <div className="search-field">
                  <Icon name="search" size={18} />
                  <input
                    aria-label="Search runs"
                    placeholder="Search video or camera name"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                  {search && (
                    <button
                      aria-label="Clear search"
                      className="icon-button"
                      onClick={() => setSearch("")}
                    >
                      <Icon name="close" size={16} />
                    </button>
                  )}
                </div>
                <Button onClick={() => setRefresh((x) => x + 1)}>
                  <Icon name="refresh" size={16} />
                  Refresh
                </Button>
              </div>
              {filteredRuns.length ? (
                <div className="table-scroll">
                  <table className="runs-table">
                    <thead>
                      <tr>
                        <th>Video / study</th>
                        <th>Venue</th>
                        <th>Created</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRuns.map((j) => (
                        <tr key={j.id}>
                          <td>
                            <div className="run-name">
                              {j.video ? (
                                <img src={j.video.poster} alt="" />
                              ) : (
                                <span className="history-symbol">
                                  <Icon name="branch" />
                                </span>
                              )}
                              <div>
                                <strong>
                                  {j.video?.name ||
                                    "Illustrative scenario study"}
                                </strong>
                                <small>
                                  {j.body.camera_name} · {j.id.slice(0, 8)}
                                </small>
                              </div>
                            </div>
                          </td>
                          <td>{j.body.venue.name}</td>
                          <td>
                            {new Date(j.created * 1000).toLocaleDateString(
                              "en-IN",
                            )}
                          </td>
                          <td>
                            <span
                              className={`badge ${j.status === "completed" ? "green" : j.status === "failed" ? "red" : "blue"}`}
                            >
                              {j.status}
                            </span>
                            {active(j) && (
                              <small className="block">
                                {Math.round(j.progress * 100)}%
                              </small>
                            )}
                          </td>
                          <td>
                            <div className="button-group">
                              <Button
                                variant="ghost"
                                onClick={() => openRun(j)}
                              >
                                Open
                                <Icon name="arrow" size={14} />
                              </Button>
                              {["failed", "cancelled"].includes(j.status) && (
                                <Button
                                  variant="ghost"
                                  onClick={() => retry(j)}
                                >
                                  Retry
                                </Button>
                              )}
                              {active(j) && (
                                <Button
                                  variant="ghost"
                                  onClick={() => cancel(j.id)}
                                >
                                  Cancel
                                </Button>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty
                  icon="video"
                  title={
                    search
                      ? "No matching runs"
                      : "Your analysis history starts here"
                  }
                  description={
                    search
                      ? "Try a different video or camera name."
                      : "Upload a video or open the illustrative scenario to explore the workspace."
                  }
                  action={
                    <Button
                      onClick={
                        search ? () => setSearch("") : () => setUploadOpen(true)
                      }
                    >
                      {search ? "Clear search" : "Analyze video"}
                    </Button>
                  }
                />
              )}
              <div className="library-footer">
                <span>
                  Showing {filteredRuns.length} of {runs.length} loaded runs
                </span>
                <Button
                  disabled={runs.length < libraryLimit}
                  onClick={() => setLibraryLimit((x) => x + 30)}
                >
                  Load more
                </Button>
              </div>
            </Panel>
          )}
          {view === "models" && (
            <>
              <div className="model-intro">
                <Icon name="shield" size={32} />
                <div>
                  <h2>Every output has a source.</h2>
                  <p>
                    Checkpoint presence, runtime verification and scientific
                    validation are different states. This console keeps them
                    separate.
                  </p>
                </div>
              </div>
              <div className="model-cards">
                <Panel
                  title="DDPF perception"
                  eyebrow="YOUR SUPPLIED MODEL"
                  action={
                    <span
                      className={`badge ${health?.checkpoints_present.ddpf ? "green" : "red"}`}
                    >
                      {health?.checkpoints_present.ddpf
                        ? "Weights available"
                        : "Weights missing"}
                    </span>
                  }
                >
                  <div className="panel-padding">
                    <h3>Swin-Tiny + full-resolution decoder</h3>
                    <p>
                      Density, head heatmap and signed subpixel offsets from{" "}
                      <strong>best_balanced_ddpfnet.pth.zip</strong>.
                    </p>
                    <dl>
                      <dt>Input</dt>
                      <dd>512 × 512 · letterbox · ImageNet normalization</dd>
                      <dt>Density scaling</dt>
                      <dd>Sum / 100 · padding excluded from venue counts</dd>
                      <dt>Runtime</dt>
                      <dd>
                        {health?.models?.device ||
                          "Verified when first analysis starts"}
                      </dd>
                      <dt>Checkpoint hash</dt>
                      <dd className="hash">
                        {health?.models?.ddpf.sha256 ||
                          "Available after model initialization"}
                      </dd>
                    </dl>
                  </div>
                </Panel>
                <Panel
                  title="Temporal & graph forecasting"
                  eyebrow="COMPATIBILITY GATED"
                  action={
                    <span
                      className={`badge ${health?.models?.temporal.ready ? "green" : "amber"}`}
                    >
                      {health?.models?.temporal.ready
                        ? "Compatible"
                        : "Training required"}
                    </span>
                  }
                >
                  <div className="panel-padding">
                    <h3>STRFE → graph reasoning</h3>
                    <p>
                      The supplied DDPF exports 256-channel features. Older
                      16-channel STRFE weights cannot be substituted.
                    </p>
                    <dl>
                      <dt>Active continuation</dt>
                      <dd>
                        {health?.models?.temporal.ready
                          ? "Compatible learned model + baseline"
                          : "Persistence baseline"}
                      </dd>
                      <dt>Model architecture</dt>
                      <dd>ConvLSTM temporal fusion + graph transformer</dd>
                      <dt>Promotion gate</dt>
                      <dd>
                        Compatible feature provenance and held-out baseline
                        comparison
                      </dd>
                    </dl>
                    <p className="micro-copy">
                      Training and evaluation tooling is included. Checkpoint
                      compatibility never implies real-world safety validation.
                    </p>
                  </div>
                </Panel>
                <Panel
                  title="Intervention engine"
                  eyebrow="PHYSICAL CONSTRAINTS"
                  action={
                    <span className="badge blue">Conditional simulation</span>
                  }
                >
                  <div className="panel-padding">
                    <h3>Conserving crowd-flow model</h3>
                    <p>
                      Directed routes, finite gate capacity, receiving-space
                      limits, outside queues, timed interventions and capacity
                      sensitivity.
                    </p>
                    <dl>
                      <dt>Closing a gate</dt>
                      <dd>Zero permitted flow through the portal</dd>
                      <dt>Population</dt>
                      <dd>Conserved across zones, exits and outside queues</dd>
                      <dt>Calibration</dt>
                      <dd>Venue-specific measurements required</dd>
                    </dl>
                  </div>
                </Panel>
              </div>
              <Status message="The application does not report calibrated emergency probabilities or claim causal validation of gate interventions. Scientific accuracy must be evaluated on independent, appropriately labelled data." />
            </>
          )}
          <footer className="page-footer">
            <span>BEYOND SURVEILLANCE</span>
            <span>Video estimates · Forecasts · Conditional scenarios</span>
            <span>Processed locally</span>
          </footer>
        </main>
      </div>
      <UploadDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        venues={venues}
        onRun={(j) => {
          openRun(j);
          notify(
            "Analysis queued. You can review completed snapshots as processing continues.",
            "success",
          );
          setRefresh((x) => x + 1);
        }}
      />
    </div>
  );
}

function historyReplace(view: string) {
  const url = new URL(location.href);
  url.searchParams.set("view", view);
  window.history.pushState({}, "", url);
}
