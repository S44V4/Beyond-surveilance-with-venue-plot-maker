import { useEffect, useRef, useState } from "react";
import type { Portal, Venue, Zone } from "./types";
import { api, post } from "./api";
import { Button, Field, Icon, Panel, Select, Status, Textarea } from "./ui";
import { VenueMap } from "./Map";

function validPolygon(value: unknown): value is [number, number][] {
  return (
    Array.isArray(value) &&
    value.length >= 3 &&
    value.length <= 40 &&
    value.every(
      (p) =>
        Array.isArray(p) &&
        p.length === 2 &&
        p.every(
          (n) =>
            typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= 1000,
        ),
    )
  );
}

export function VenueEditor({
  venues,
  onSave,
}: {
  venues: Venue[];
  onSave: (v: Venue) => void;
}) {
  const [draft, setDraft] = useState<Venue | null>(null),
    [zone, setZone] = useState(""),
    [portal, setPortal] = useState(""),
    [error, setError] = useState(""),
    [success, setSuccess] = useState(""),
    [busy, setBusy] = useState(false),
    [json, setJson] = useState(""),
    [polygon, setPolygon] = useState(""),
    [imagePolygon, setImagePolygon] = useState(""),
    [drawing, setDrawing] = useState(false),
    [points, setPoints] = useState<number[][]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!draft && venues.length) {
      let restored: Venue | undefined;
      try {
        restored =
          JSON.parse(localStorage.getItem("bs-venue-draft") || "null") ||
          undefined;
        if (
          restored &&
          (!Array.isArray(restored.zones) ||
            !restored.zones.length ||
            !Array.isArray(restored.portals) ||
            restored.zones.some((z) => !validPolygon(z.polygon)))
        )
          restored = undefined;
      } catch {
        /* invalid local draft ignored */
      }
      setDraft(restored || structuredClone(venues[0]));
      setZone((restored || venues[0]).zones[0].id);
    }
  }, [venues, draft]);
  useEffect(() => {
    if (draft) localStorage.setItem("bs-venue-draft", JSON.stringify(draft));
  }, [draft]);
  const selected = draft?.zones.find((z) => z.id === zone),
    gate = draft?.portals.find((p) => p.id === portal);
  useEffect(() => {
    setPolygon(JSON.stringify(selected?.polygon || []));
    setImagePolygon(
      JSON.stringify(selected?.image_polygon || selected?.polygon || []),
    );
  }, [zone, draft?.id]);
  function updateZone(change: Partial<Zone>) {
    if (!draft) return;
    setDraft({
      ...draft,
      zones: draft.zones.map((z) => (z.id === zone ? { ...z, ...change } : z)),
    });
    setSuccess("");
  }
  function updatePortal(change: Partial<Portal>) {
    if (!draft) return;
    setDraft({
      ...draft,
      portals: draft.portals.map((p) =>
        p.id === portal ? { ...p, ...change } : p,
      ),
    });
    setSuccess("");
  }
  function useVenue(v: Venue) {
    setDraft(structuredClone(v));
    setZone(v.zones[0].id);
    setPortal("");
    setError("");
    setSuccess("");
    setJson("");
    setDrawing(false);
  }
  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!draft) return;
    setBusy(true);
    setError("");
    try {
      const saved = await post<Venue>("/venues", draft);
      setDraft(saved);
      onSave(saved);
      setSuccess(`Venue saved · version ${saved.version}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function applyGeometry() {
    try {
      const p = JSON.parse(polygon),
        ip = JSON.parse(imagePolygon);
      if (!validPolygon(p) || !validPolygon(ip))
        throw new Error("Use at least three [x,y] coordinate pairs.");
      updateZone({ polygon: p, image_polygon: ip });
      setError("");
    } catch (e) {
      setError(`Invalid polygon: ${(e as Error).message}`);
    }
  }
  async function uploadBackground(file?: File) {
    if (!file || !draft) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const result = await api<{ url: string }>("/assets", {
        method: "POST",
        body: form,
      });
      setDraft({ ...draft, background: result.url });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function addZone() {
    if (!draft) return;
    const id = `Z${Date.now().toString(36)}`;
    const next: Zone = {
      id,
      name: `Zone ${draft.zones.length + 1}`,
      polygon: [
        [100, 100],
        [350, 100],
        [350, 350],
        [100, 350],
      ],
      image_polygon: [
        [100, 100],
        [350, 100],
        [350, 350],
        [100, 350],
      ],
      capacity: 100,
      area_m2: null,
      observed: true,
    };
    setDraft({ ...draft, zones: [...draft.zones, next] });
    setZone(id);
  }
  function addPortal() {
    if (!draft) return;
    const id = `G${Date.now().toString(36)}`;
    setDraft({
      ...draft,
      portals: [
        ...draft.portals,
        {
          id,
          name: "New exit",
          source: draft.zones[0].id,
          target: "outside",
          kind: "exit",
          position: [500, 70],
          capacity: 1.5,
          open: true,
          bidirectional: false,
          travel_seconds: 5,
        },
      ],
    });
    setPortal(id);
  }
  if (!draft)
    return (
      <Panel>
        <p className="panel-padding">Loading venue definitions…</p>
      </Panel>
    );
  return (
    <form noValidate onSubmit={save}>
      <div className="setup-toolbar">
        <div>
          <h2>Make the map match your space.</h2>
          <p>
            Define zones, routes and gates. Each saved run keeps its own venue
            version.
          </p>
        </div>
        <div className="button-group">
          <Button
            type="button"
            onClick={() => {
              const copy = structuredClone(draft);
              copy.id = `venue-${Date.now().toString(36)}`;
              copy.name = "New venue";
              copy.version = 1;
              useVenue(copy);
            }}
          >
            <Icon name="plus" size={17} />
            New from this layout
          </Button>
          <Button variant="primary" type="submit" busy={busy}>
            <Icon name="check" size={17} />
            Save venue
          </Button>
        </div>
      </div>
      {error && <Status message={error} tone="error" />}
      {success && <Status message={success} tone="success" />}
      <div className="setup-grid">
        <Panel
          title="Venue layout"
          action={<span className="badge">Draft saved locally</span>}
        >
          <div className="setup-map-wrap">
            <VenueMap
              venue={draft}
              selected={zone}
              onSelect={setZone}
              onGate={setPortal}
            />
            {drawing && (
              <svg
                className="draw-overlay"
                viewBox="0 0 1000 960"
                aria-label="Click map to add polygon points"
                onClick={(e) => {
                  const matrix = e.currentTarget.getScreenCTM();
                  if (!matrix) return;
                  const point = new DOMPoint(
                    e.clientX,
                    e.clientY,
                  ).matrixTransform(matrix.inverse());
                  setPoints([
                    ...points,
                    [
                      Math.max(0, Math.min(1000, point.x)),
                      Math.max(0, Math.min(1000, point.y)),
                    ],
                  ]);
                }}
              >
                <polyline
                  points={points.map((p) => p.join(",")).join(" ")}
                  fill="rgba(36,86,214,.12)"
                  stroke="#2456d6"
                  strokeWidth="3"
                />
                {points.map((p, i) => (
                  <circle key={i} cx={p[0]} cy={p[1]} r="6" fill="#2456d6" />
                ))}
              </svg>
            )}
          </div>
          <div className="map-edit-toolbar">
            <Button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={busy}
            >
              <Icon name="upload" size={16} />
              Floor plan
            </Button>
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg"
              className="sr-only"
              aria-label="Floor plan image"
              onChange={(e) => uploadBackground(e.target.files?.[0])}
            />
            <Button
              type="button"
              onClick={() => {
                setDrawing(!drawing);
                setPoints([]);
              }}
            >
              {drawing ? "Cancel drawing" : "Draw selected zone"}
            </Button>
            {drawing && (
              <Button
                type="button"
                disabled={points.length < 3}
                onClick={() => {
                  updateZone({ polygon: points });
                  setPolygon(JSON.stringify(points));
                  setDrawing(false);
                }}
              >
                Finish ({points.length} points)
              </Button>
            )}
          </div>
          <p className="panel-note">
            Drawing changes the floor-plan polygon. Set the camera-image polygon
            separately below. Coordinates can also be typed; every control works
            without dragging.
          </p>
        </Panel>
        <div className="setup-fields">
          <Panel title="Venue details">
            <div className="panel-padding">
              <Field label="Existing venue" id="existing-venue">
                <Select
                  id="existing-venue"
                  value={venues.some((v) => v.id === draft.id) ? draft.id : ""}
                  onChange={(e) => {
                    const v = venues.find((v) => v.id === e.target.value);
                    if (v) useVenue(v);
                  }}
                >
                  <option value="" disabled>
                    Unsaved venue
                  </option>
                  {venues.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.name}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Venue name" id="venue-name">
                <input
                  id="venue-name"
                  value={draft.name}
                  maxLength={100}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                />
              </Field>
              <Field label="Geometry evidence" id="geometry">
                <Select
                  id="geometry"
                  value={draft.geometry_kind}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      geometry_kind: e.target.value as Venue["geometry_kind"],
                    })
                  }
                >
                  <option value="schematic">Schematic / assumed layout</option>
                  <option value="reviewed_image_zones">
                    Reviewed image zones
                  </option>
                  <option value="calibrated">Calibrated physical venue</option>
                </Select>
              </Field>
              <Field label="Description" id="description">
                <Textarea
                  className="resize-none"
                  id="description"
                  rows={3}
                  value={draft.description}
                  onChange={(e) =>
                    setDraft({ ...draft, description: e.target.value })
                  }
                />
              </Field>
              <Field label="Calibration evidence / units" id="calibration">
                <Textarea
                  className="resize-none"
                  id="calibration"
                  rows={2}
                  placeholder="Reference measurements and mapping method"
                  value={draft.calibration_note}
                  onChange={(e) =>
                    setDraft({ ...draft, calibration_note: e.target.value })
                  }
                />
              </Field>
            </div>
          </Panel>
          <Panel
            title="Selected zone"
            action={
              <Button type="button" variant="ghost" onClick={addZone}>
                <Icon name="plus" size={16} />
                Add
              </Button>
            }
          >
            <div className="panel-padding">
              <Field label="Zone" id="zone-select">
                <Select
                  id="zone-select"
                  value={zone}
                  onChange={(e) => setZone(e.target.value)}
                >
                  {draft.zones.map((z) => (
                    <option key={z.id} value={z.id}>
                      {z.id} · {z.name}
                    </option>
                  ))}
                </Select>
              </Field>
              {selected && (
                <>
                  <Field label="Zone name" id="zone-name">
                    <input
                      id="zone-name"
                      value={selected.name}
                      onChange={(e) => updateZone({ name: e.target.value })}
                    />
                  </Field>
                  <div className="form-grid">
                    <Field label="Occupancy capacity" id="zone-capacity">
                      <input
                        id="zone-capacity"
                        type="number"
                        min="1"
                        value={selected.capacity}
                        onChange={(e) =>
                          updateZone({ capacity: Number(e.target.value) })
                        }
                      />
                    </Field>
                    <Field label="Measured area (m²)" id="zone-area">
                      <input
                        id="zone-area"
                        type="number"
                        min=".1"
                        placeholder="Unknown"
                        value={selected.area_m2 ?? ""}
                        onChange={(e) =>
                          updateZone({
                            area_m2:
                              e.target.value === ""
                                ? null
                                : Number(e.target.value),
                          })
                        }
                      />
                    </Field>
                  </div>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={selected.observed}
                      onChange={(e) =>
                        updateZone({ observed: e.target.checked })
                      }
                    />
                    Visible in this camera
                  </label>
                  <details>
                    <summary>Map & camera coordinates</summary>
                    <div className="details-content">
                      <Field label="Map polygon [x,y], 0–1000" id="map-polygon">
                        <Textarea
                          id="map-polygon"
                          className="code-input resize-none"
                          rows={3}
                          value={polygon}
                          onChange={(e) => setPolygon(e.target.value)}
                        />
                      </Field>
                      <Field
                        label="Camera polygon [x,y], 0–1000"
                        id="image-polygon"
                      >
                        <Textarea
                          id="image-polygon"
                          className="code-input resize-none"
                          rows={3}
                          value={imagePolygon}
                          onChange={(e) => setImagePolygon(e.target.value)}
                        />
                      </Field>
                      <Button type="button" onClick={applyGeometry}>
                        Apply coordinates
                      </Button>
                    </div>
                  </details>
                </>
              )}
            </div>
          </Panel>
        </div>
      </div>
      <div className="setup-bottom">
        <Panel
          title="Portals & movement"
          action={
            <Button type="button" onClick={addPortal}>
              <Icon name="plus" size={16} />
              Add portal
            </Button>
          }
        >
          <div className="panel-padding">
            <Field label="Portal" id="portal-select">
              <Select
                id="portal-select"
                value={portal}
                onChange={(e) => setPortal(e.target.value)}
              >
                <option value="">Choose a gate or passage</option>
                {draft.portals.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </Select>
            </Field>
            {gate && (
              <>
                <div className="form-grid">
                  <Field label="Name" id="portal-name">
                    <input
                      id="portal-name"
                      value={gate.name}
                      onChange={(e) => updatePortal({ name: e.target.value })}
                    />
                  </Field>
                  <Field label="Type" id="portal-kind">
                    <Select
                      id="portal-kind"
                      value={gate.kind}
                      onChange={(e) => {
                        const kind = e.target.value as Portal["kind"];
                        updatePortal({
                          kind,
                          source:
                            kind === "entrance" ? "outside" : draft.zones[0].id,
                          target:
                            kind === "exit"
                              ? "outside"
                              : draft.zones[
                                  kind === "passage"
                                    ? Math.min(1, draft.zones.length - 1)
                                    : 0
                                ].id,
                          bidirectional: kind === "passage",
                        });
                      }}
                    >
                      <option value="entrance">Entrance</option>
                      <option value="exit">Exit</option>
                      <option value="passage">Internal passage</option>
                    </Select>
                  </Field>
                </div>
                <div className="form-grid">
                  <Field label="From" id="from">
                    <Select
                      id="from"
                      value={gate.source}
                      onChange={(e) => updatePortal({ source: e.target.value })}
                    >
                      <option value="outside">Outside</option>
                      {draft.zones.map((z) => (
                        <option key={z.id} value={z.id}>
                          {z.name}
                        </option>
                      ))}
                    </Select>
                  </Field>
                  <Field label="To" id="to">
                    <Select
                      id="to"
                      value={gate.target}
                      onChange={(e) => updatePortal({ target: e.target.value })}
                    >
                      <option value="outside">Outside</option>
                      {draft.zones.map((z) => (
                        <option key={z.id} value={z.id}>
                          {z.name}
                        </option>
                      ))}
                    </Select>
                  </Field>
                </div>
                <div className="form-grid">
                  <Field label="Capacity (people/s)" id="portal-cap">
                    <input
                      id="portal-cap"
                      type="number"
                      step=".1"
                      value={gate.capacity}
                      onChange={(e) =>
                        updatePortal({ capacity: Number(e.target.value) })
                      }
                    />
                  </Field>
                  <Field label="Travel time (s)" id="travel">
                    <input
                      id="travel"
                      type="number"
                      value={gate.travel_seconds}
                      onChange={(e) =>
                        updatePortal({ travel_seconds: Number(e.target.value) })
                      }
                    />
                  </Field>
                </div>
                <div className="form-grid">
                  <Field label="Map X" id="portal-x">
                    <input
                      id="portal-x"
                      type="number"
                      min="0"
                      max="1000"
                      value={gate.position[0]}
                      onChange={(e) =>
                        updatePortal({
                          position: [Number(e.target.value), gate.position[1]],
                        })
                      }
                    />
                  </Field>
                  <Field label="Map Y" id="portal-y">
                    <input
                      id="portal-y"
                      type="number"
                      min="0"
                      max="1000"
                      value={gate.position[1]}
                      onChange={(e) =>
                        updatePortal({
                          position: [gate.position[0], Number(e.target.value)],
                        })
                      }
                    />
                  </Field>
                </div>
                <div className="button-group">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={gate.open}
                      onChange={(e) => updatePortal({ open: e.target.checked })}
                    />
                    Open in baseline
                  </label>
                  {gate.kind === "passage" && (
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={gate.bidirectional}
                        onChange={(e) =>
                          updatePortal({ bidirectional: e.target.checked })
                        }
                      />
                      Bidirectional
                    </label>
                  )}
                </div>
              </>
            )}
          </div>
        </Panel>
        <Panel title="Import & advanced editing">
          <div className="panel-padding">
            <p className="muted">
              Use the versioned JSON schema for arbitrary polygons, removing
              zones or portals, and transferring a reviewed venue between
              machines.
            </p>
            <div className="button-group">
              <Button
                type="button"
                onClick={() => setJson(JSON.stringify(draft, null, 2))}
              >
                <Icon name="file" size={17} />
                Load draft JSON
              </Button>
              <Button
                type="button"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(draft, null, 2)], {
                    type: "application/json",
                  });
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(blob);
                  a.download = `${draft.id}.json`;
                  a.click();
                  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
                }}
              >
                <Icon name="download" size={17} />
                Export JSON
              </Button>
            </div>
            <Field label="Venue JSON" id="venue-json">
              <Textarea
                id="venue-json"
                rows={14}
                className="code-input resize-none"
                value={json}
                onChange={(e) => setJson(e.target.value)}
                placeholder="Load the draft JSON or paste a venue definition"
              />
            </Field>
            <Button
              type="button"
              disabled={!json.trim()}
              onClick={async () => {
                try {
                  const value = await post<Venue>(
                    "/venues/validate",
                    JSON.parse(json),
                  );
                  if (
                    !value.id ||
                    !value.zones?.length ||
                    !Array.isArray(value.portals)
                  )
                    throw new Error(
                      "Expected a venue with id, zones and portals.",
                    );
                  useVenue(value);
                  setSuccess(
                    "JSON applied to draft. Save to validate and persist it.",
                  );
                } catch (e) {
                  setError((e as Error).message);
                }
              }}
            >
              Apply JSON to draft
            </Button>
            <p className="micro-copy">
              Saving validates geometry, portal endpoints, unique identifiers
              and physical-unit requirements on the server.
            </p>
          </div>
        </Panel>
      </div>
    </form>
  );
}
