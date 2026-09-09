import { useEffect, useState } from "react";
import type { Job, ScenarioResult } from "./types";
import { Button, Empty, Icon, Panel, Status } from "./ui";
import { VenueMap } from "./Map";
import { Chart } from "./Chart";
import { clock, number } from "./api";
export function ScenarioView({
  job,
  result,
  onBack,
  onCancel,
}: {
  job: Job | null;
  result: ScenarioResult | null;
  onBack: () => void;
  onCancel: () => void;
}) {
  const [time, setTime] = useState(0),
    [play, setPlay] = useState(false),
    [zone, setZone] = useState("");
  useEffect(() => {
    setTime(0);
    setPlay(false);
  }, [result?.id]);
  useEffect(() => {
    if (!play || !result) return;
    const timer = setInterval(
      () =>
        setTime((t) => {
          if (t >= result.request.duration) {
            setPlay(false);
            return t;
          }
          return Math.min(t + 1, result.request.duration);
        }),
      80,
    );
    return () => clearInterval(timer);
  }, [play, result]);
  if (!result)
    return (
      <Panel>
        <Empty
          icon="branch"
          title={
            job && ["queued", "running"].includes(job.status)
              ? "Simulating crowd movement…"
              : "A clearer view of what could change"
          }
          description={
            job?.error ||
            "Choose a video snapshot or an illustrative state, change the gates, and compare both futures from the same starting point."
          }
          action={
            <Button
              onClick={
                job && ["queued", "running"].includes(job.status)
                  ? onCancel
                  : onBack
              }
            >
              {job && ["queued", "running"].includes(job.status)
                ? "Cancel simulation"
                : "Configure a scenario"}
            </Button>
          }
        />
      </Panel>
    );
  const b =
      result.baseline.trajectory[
        Math.min(time, result.baseline.trajectory.length - 1)
      ],
    s =
      result.scenario.trajectory[
        Math.min(time, result.scenario.trajectory.length - 1)
      ];
  const metrics = [
    [
      "People remaining",
      result.baseline.metrics.remaining,
      result.scenario.metrics.remaining,
    ],
    [
      "Departed",
      result.baseline.metrics.departed,
      result.scenario.metrics.departed,
    ],
    [
      "Outside queue",
      result.baseline.metrics.outside_queue,
      result.scenario.metrics.outside_queue,
    ],
    [
      "Without an exit route",
      result.baseline.metrics.unreachable,
      result.scenario.metrics.unreachable,
    ],
  ] as const;
  return (
    <>
      <div className="scenario-title">
        <div>
          <span className="eyebrow">SCENARIO / {result.id.slice(0, 8)}</span>
          <h2>{result.name}</h2>
          <p>
            Snapshot at {clock(result.source_timestamp)} ·{" "}
            {result.request.mode === "continuous"
              ? "Continuous arrivals"
              : "Arrivals stopped"}{" "}
            · {result.request.duration}s horizon
          </p>
        </div>
        <Button onClick={onBack}>
          <Icon name="settings" size={17} />
          Configure another
        </Button>
      </div>
      <Status
        tone="info"
        message={
          result.source_evidence === "hypothetical_scenario_input"
            ? "Illustrative study: initial crowd counts, layout, and gate capacities are assumptions."
            : "Conditional simulation initialized from video estimates. Gate capacities and route choices remain assumptions."
        }
      />
      <div className="comparison-maps">
        <Panel
          title="Baseline"
          eyebrow="UNCHANGED CONFIGURATION"
          action={<span className="badge blue">{number(b.total)} people</span>}
        >
          <VenueMap
            venue={result.venue}
            counts={b.counts}
            gateOpen={b.gate_open}
            selected={zone}
            onSelect={setZone}
            compact
          />
        </Panel>
        <Panel
          title="Your scenario"
          eyebrow="MODIFIED CONFIGURATION"
          action={<span className="badge amber">{number(s.total)} people</span>}
        >
          <VenueMap
            venue={result.venue}
            counts={s.counts}
            gateOpen={s.gate_open}
            changes={result.request.gates}
            selected={zone}
            onSelect={setZone}
            compact
          />
        </Panel>
      </div>
      <div className="playback scenario-playback">
        <button
          className="play-button"
          aria-label={play ? "Pause simulation" : "Play simulation"}
          onClick={() => {
            if (time >= result.request.duration) setTime(0);
            setPlay(!play);
          }}
        >
          <Icon name={play ? "pause" : "play"} size={16} />
        </button>
        <span className="mono">+{clock(time)}</span>
        <input
          type="range"
          min="0"
          max={result.request.duration}
          value={time}
          aria-label="Simulation time"
          onChange={(e) => {
            setTime(Number(e.target.value));
            setPlay(false);
          }}
        />
        <span className="mono">{clock(result.request.duration)}</span>
        <span className="badge">12.5× playback</span>
      </div>
      <div className="comparison-bottom">
        <Panel
          title="How the crowd changes"
          action={<span className="muted">People inside the venue</span>}
        >
          <Chart
            series={[
              {
                name: "Baseline",
                values: result.baseline.trajectory.map((t) => t.total),
                color: "#2456d6",
              },
              {
                name: "Your scenario",
                values: result.scenario.trajectory.map((t) => t.total),
                color: "#d87736",
              },
            ]}
            labels={[
              "Now",
              `+${clock(result.request.duration / 2)}`,
              `+${clock(result.request.duration)}`,
            ]}
            height={210}
          />
          {zone && (
            <div className="zone-comparison">
              <strong>
                {result.venue.zones.find((z) => z.id === zone)?.name}
              </strong>
              <span>Baseline {number(b.counts[zone], 1)}</span>
              <span>Scenario {number(s.counts[zone], 1)}</span>
            </div>
          )}
        </Panel>
        <Panel title="At the end of the horizon">
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Measure</th>
                  <th>Baseline</th>
                  <th>Scenario</th>
                  <th>Change</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map(([label, a, b]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{number(a, 1)}</td>
                    <td>{number(b, 1)}</td>
                    <td className={b > a ? "delta-warm" : "delta-cool"}>
                      {b - a > 0 ? "+" : ""}
                      {number(b - a, 1)}
                    </td>
                  </tr>
                ))}
                <tr>
                  <td>Clearance time</td>
                  <td>
                    {result.baseline.metrics.clearance_seconds === null
                      ? "Not cleared"
                      : `${number(result.baseline.metrics.clearance_seconds)}s`}
                  </td>
                  <td>
                    {result.scenario.metrics.clearance_seconds === null
                      ? "Not cleared"
                      : `${number(result.scenario.metrics.clearance_seconds)}s`}
                  </td>
                  <td>—</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
      <div className="comparison-bottom">
        <Panel title="Exit throughput">
          <div className="throughput-list">
            {result.venue.portals
              .filter((p) => p.kind === "exit")
              .map((p) => (
                <div key={p.id}>
                  <span>
                    <Icon name="gate" size={17} />
                    {p.name}
                  </span>
                  <strong>
                    {number(result.scenario.metrics.gate_totals[p.id], 1)}{" "}
                    <small>departed</small>
                  </strong>
                  <div className="throughput-track">
                    <i
                      style={{
                        width: `${Math.min(100, (100 * result.scenario.metrics.gate_totals[p.id]) / Math.max(1, result.scenario.metrics.initial))}%`,
                      }}
                    />
                  </div>
                  <small>
                    Baseline{" "}
                    {number(result.baseline.metrics.gate_totals[p.id], 1)}
                  </small>
                </div>
              ))}
          </div>
        </Panel>
        <Panel title="Assumptions & sensitivity">
          <div className="assumptions">
            <p>
              Conservation residual:{" "}
              <strong>
                {result.scenario.metrics.conservation_error.toExponential(2)}{" "}
                people
              </strong>
            </p>
            {result.sensitivity.map((x) => (
              <div className="sensitivity-row" key={x.capacity_factor}>
                <span>Capacity {Math.round(x.capacity_factor * 100)}%</span>
                <span>{number(x.scenario_remaining, 1)} remaining</span>
                <span>Δ {number(x.remaining_delta, 1)}</span>
              </div>
            ))}
            <details>
              <summary>Read simulation assumptions</summary>
              <ul>
                {result.assumptions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </details>
          </div>
        </Panel>
      </div>
    </>
  );
}
