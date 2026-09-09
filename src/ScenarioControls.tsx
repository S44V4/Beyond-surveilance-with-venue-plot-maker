import { useState } from "react";
import type { GateChanges, Job, ScenarioInput, Snapshot, Venue } from "./types";
import { Button, Field, Icon, Select, Status } from "./ui";
import { idempotency, number } from "./api";
export function ScenarioControls({
  venue,
  snapshot,
  run,
  changes,
  setChanges,
  onRun,
  busy,
}: {
  venue: Venue;
  snapshot: Snapshot | null;
  run: Job | null;
  changes: GateChanges;
  setChanges: (g: GateChanges) => void;
  onRun: (input: ScenarioInput) => void;
  busy: boolean;
}) {
  const [duration, setDuration] = useState("120"),
    [start, setStart] = useState("0"),
    [end, setEnd] = useState(""),
    [mode, setMode] = useState<"evacuation" | "continuous">("evacuation"),
    [inflow, setInflow] = useState("1"),
    [name, setName] = useState("Gate configuration"),
    [error, setError] = useState(""),
    [unobserved, setUnobserved] = useState<Record<string, number>>({});
  const gates = venue.portals.filter((p) => p.kind !== "passage");
  const count = Object.keys(changes).length;
  function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!run || !snapshot) return;
    const d = Number(duration),
      s = Number(start),
      until = end === "" ? null : Number(end);
    if (
      !name.trim() ||
      !Number.isFinite(d) ||
      d < 10 ||
      d > 600 ||
      s < 0 ||
      s >= d ||
      (until !== null && (until <= s || until > d))
    ) {
      setError(
        "Use a name, a 10–600 second horizon, and intervention times inside that horizon.",
      );
      return;
    }
    onRun({
      run_id: run.id,
      snapshot_index: snapshot.index,
      name: name.trim(),
      gates: changes,
      duration: d,
      start: s,
      end: until,
      mode,
      inflow_per_second: mode === "continuous" ? Number(inflow) : 0,
      unobserved_counts: unobserved,
      sensitivity: true,
      idempotency_key: idempotency(),
    });
  }
  return (
    <form noValidate onSubmit={submit} className="scenario-controls">
      <div className="inspector-intro">
        <span className="small-icon">
          <Icon name="branch" />
        </span>
        <div>
          <h2>What if?</h2>
          <p>Explore a different gate configuration.</p>
        </div>
      </div>
      <div className="inspector-section">
        <div className="section-label">
          <span>Entrances & exits</span>
          <button
            type="button"
            className="text-button"
            disabled={!count || busy}
            onClick={() => setChanges({})}
          >
            Reset
          </button>
        </div>
        <div className="gate-list">
          {gates.map((p) => {
            const open = changes[p.id]?.open ?? p.open;
            return (
              <div
                className={`gate-row ${changes[p.id] ? "changed" : ""}`}
                key={p.id}
              >
                <span className={`gate-icon ${open ? "open" : "closed"}`}>
                  <Icon name="gate" size={18} />
                </span>
                <div>
                  <strong>{p.name}</strong>
                  <span>
                    {p.kind === "entrance"
                      ? "Arrival point"
                      : "Departure point"}{" "}
                    · {number(changes[p.id]?.capacity ?? p.capacity, 1)} p/s
                  </span>
                </div>
                <button
                  type="button"
                  className={`switch ${open ? "on" : ""}`}
                  role="switch"
                  aria-checked={open}
                  aria-label={`${p.name} open`}
                  disabled={busy}
                  onClick={() =>
                    setChanges({
                      ...changes,
                      [p.id]: { ...changes[p.id], open: !open },
                    })
                  }
                >
                  <i />
                </button>
                <small className={open ? "open-text" : "closed-text"}>
                  {open ? "Open" : "Closed"}
                </small>
              </div>
            );
          })}
        </div>
        <p className="micro-copy">
          Changes apply to this simulation. Recorded footage and the baseline
          stay intact.
        </p>
      </div>
      <div className="inspector-section">
        <Field label="Scenario name" id="scenario-name">
          <input
            id="scenario-name"
            value={name}
            maxLength={100}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <div className="form-grid">
          <Field label="Horizon" id="duration">
            <Select
              id="duration"
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
            >
              <option value="60">60 seconds</option>
              <option value="120">120 seconds</option>
              <option value="300">5 minutes</option>
              <option value="600">10 minutes</option>
            </Select>
          </Field>
          <Field label="Demand mode" id="mode">
            <Select
              id="mode"
              value={mode}
              onChange={(e) => setMode(e.target.value as typeof mode)}
            >
              <option value="evacuation">Stop arrivals</option>
              <option value="continuous">Continue arrivals</option>
            </Select>
          </Field>
        </div>
        <details>
          <summary>Timing, capacity & assumptions</summary>
          <div className="details-content">
            <div className="form-grid">
              <Field label="Start after (s)" id="start">
                <input
                  id="start"
                  type="number"
                  min="0"
                  max={Number(duration) - 1}
                  value={start}
                  onChange={(e) => setStart(e.target.value)}
                />
              </Field>
              <Field label="Restore after (s)" id="end">
                <input
                  id="end"
                  type="number"
                  placeholder="Until end"
                  value={end}
                  onChange={(e) => setEnd(e.target.value)}
                />
              </Field>
            </div>
            {mode === "continuous" && (
              <Field label="Total outside arrivals (people/s)" id="inflow">
                <input
                  id="inflow"
                  type="number"
                  min="0"
                  max="100"
                  step=".1"
                  value={inflow}
                  onChange={(e) => setInflow(e.target.value)}
                />
              </Field>
            )}
            {venue.portals.map((p) => (
              <div className="capacity-row" key={p.id}>
                <label htmlFor={`cap-${p.id}`}>{p.name}</label>
                <input
                  id={`cap-${p.id}`}
                  type="number"
                  min=".1"
                  max="500"
                  step=".1"
                  value={changes[p.id]?.capacity ?? p.capacity}
                  onChange={(e) =>
                    setChanges({
                      ...changes,
                      [p.id]: {
                        open: changes[p.id]?.open ?? p.open,
                        capacity: Number(e.target.value),
                      },
                    })
                  }
                />
                <button
                  type="button"
                  className={`badge-button ${(changes[p.id]?.open ?? p.open) ? "green" : "red"}`}
                  onClick={() =>
                    setChanges({
                      ...changes,
                      [p.id]: {
                        ...changes[p.id],
                        open: !(changes[p.id]?.open ?? p.open),
                      },
                    })
                  }
                >
                  {(changes[p.id]?.open ?? p.open) ? "Open" : "Closed"}
                </button>
              </div>
            ))}
            {venue.zones
              .filter((z) => !z.observed)
              .map((z) => (
                <Field
                  key={z.id}
                  label={`${z.name}: assumed initial count`}
                  id={`assumed-${z.id}`}
                >
                  <input
                    id={`assumed-${z.id}`}
                    type="number"
                    min="0"
                    value={unobserved[z.id] ?? ""}
                    onChange={(e) =>
                      setUnobserved({
                        ...unobserved,
                        [z.id]: Number(e.target.value),
                      })
                    }
                  />
                </Field>
              ))}
            <p className="micro-copy">
              Includes a ±20% capacity sensitivity sweep. Capacities are
              assumptions unless measured. Stop-arrivals mode estimates
              clearance; continuous mode models entrance queues.
            </p>
          </div>
        </details>
      </div>
      <div className="inspector-footer">
        <div className="source-line">
          <Icon name="clock" size={14} />
          {snapshot
            ? `Source snapshot · ${snapshot.timestamp.toFixed(1)} s`
            : "Choose an analyzed snapshot"}
          <span>{count} changes</span>
        </div>
        {error && <Status message={error} tone="error" />}
        <Button
          variant="primary"
          type="submit"
          className="full"
          disabled={!snapshot || !run}
          busy={busy}
        >
          <Icon name="branch" size={17} />
          Run comparison
          <Icon name="arrow" size={17} />
        </Button>
        <small>Conditional simulation · no physical gate control</small>
      </div>
    </form>
  );
}
