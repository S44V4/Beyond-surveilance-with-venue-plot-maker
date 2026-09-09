import { useId } from "react";
import type { Venue, ZoneState, GateChanges } from "./types";
import { number } from "./api";
type Props = {
  venue: Venue;
  states?: ZoneState[];
  counts?: Record<string, number>;
  changes?: GateChanges;
  gateOpen?: Record<string, boolean>;
  selected?: string;
  onSelect?: (id: string) => void;
  onGate?: (id: string) => void;
  compact?: boolean;
  heat?: boolean;
  connections?: boolean;
};
export function VenueMap({
  venue,
  states,
  counts,
  changes = {},
  gateOpen,
  selected,
  onSelect,
  onGate,
  compact = false,
  heat = true,
  connections = true,
}: Props) {
  const id = useId().replace(/:/g, "");
  const centers = Object.fromEntries(
    venue.zones.map((z) => [
      z.id,
      [
        z.polygon.reduce((s, p) => s + p[0], 0) / z.polygon.length,
        z.polygon.reduce((s, p) => s + p[1], 0) / z.polygon.length,
      ],
    ]),
  );
  const values =
    counts || Object.fromEntries((states || []).map((z) => [z.id, z.count]));
  return (
    <div className={`venue-map ${compact ? "compact" : ""}`}>
      <svg
        viewBox="0 0 1000 960"
        role="img"
        aria-label={`${venue.name}: zone occupancy and entrances and exits`}
      >
        <defs>
          <pattern
            id={`dots-${id}`}
            width="22"
            height="22"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="1" cy="1" r=".8" fill="#c6d1e2" />
          </pattern>
          <marker
            id={`arrow-${id}`}
            markerWidth="7"
            markerHeight="7"
            refX="5"
            refY="3"
            orient="auto"
          >
            <path
              d="M0 0 6 3 0 6"
              fill="none"
              stroke="#698fce"
              strokeWidth="1.3"
            />
          </marker>
        </defs>
        <rect width="1000" height="960" fill={`url(#dots-${id})`} />
        {venue.background && (
          <image
            href={venue.background}
            width="1000"
            height="960"
            preserveAspectRatio="none"
            opacity=".5"
          />
        )}
        <path
          d="M55 56h865v831H55Z"
          fill="none"
          stroke="#cad4e3"
          strokeWidth="2"
          strokeDasharray="8 5"
        />
        <text x="72" y="36" className="map-meta">
          {venue.geometry_kind === "calibrated"
            ? "CALIBRATED VENUE"
            : "SCHEMATIC · NOT TO SCALE"}
        </text>
        <g transform="translate(935,30)">
          <path d="m0 20 8-20 8 20-8-5Z" fill="#8092ad" />
          <text x="8" y="-9" textAnchor="middle" className="map-meta">
            N
          </text>
        </g>
        {connections &&
          venue.portals
            .filter((p) => p.kind === "passage")
            .map((p) => {
              const a = centers[p.source],
                b = centers[p.target];
              if (!a || !b) return null;
              const opened = gateOpen?.[p.id] ?? changes[p.id]?.open ?? p.open;
              return (
                <path
                  key={p.id}
                  d={`M${a[0]} ${a[1]}L${b[0]} ${b[1]}`}
                  stroke={opened ? "#bacce7" : "#c45662"}
                  strokeWidth={opened ? 4 : 2}
                  strokeDasharray="7 7"
                />
              );
            })}
        {venue.zones.map((z) => {
          const c = centers[z.id],
            n = values[z.id],
            ratio = n == null ? 0 : n / z.capacity;
          const state = states?.find((s) => s.id === z.id);
          return (
            <g
              key={z.id}
              className={onSelect ? "map-zone clickable" : "map-zone"}
              onClick={() => onSelect?.(z.id)}
            >
              <polygon
                points={z.polygon.map((p) => p.join(",")).join(" ")}
                fill={
                  !z.observed
                    ? "#e9edf2"
                    : heat && n != null
                      ? `hsl(219 68% ${Math.max(68, 96 - ratio * 23)}%)`
                      : "#f9fbfe"
                }
                stroke={selected === z.id ? "#2456d6" : "#aebfd8"}
                strokeWidth={selected === z.id ? 3 : 1.5}
              />
              <text
                x={c[0]}
                y={c[1] - 40}
                className="map-zone-id"
                textAnchor="middle"
              >
                {z.id}
              </text>
              <text
                x={c[0]}
                y={c[1] + 3}
                className="map-count"
                textAnchor="middle"
              >
                {number(n)}
              </text>
              <text
                x={c[0]}
                y={c[1] + 32}
                className="map-zone-name"
                textAnchor="middle"
              >
                {z.name.length > 22 ? z.name.slice(0, 20) + "…" : z.name}
              </text>
              {state?.direction && (
                <path
                  d={`M${c[0] - 18} ${c[1] + 58}l${Math.max(-30, Math.min(30, state.direction[0] * 5))} ${Math.max(-20, Math.min(20, state.direction[1] * 5))}`}
                  stroke="#507ec7"
                  strokeWidth="2"
                  markerEnd={`url(#arrow-${id})`}
                />
              )}
            </g>
          );
        })}
        {venue.portals
          .filter((p) => p.kind !== "passage")
          .map((p) => {
            const opened = gateOpen?.[p.id] ?? changes[p.id]?.open ?? p.open;
            const changed = changes[p.id] !== undefined;
            const [x, y] = p.position;
            return (
              <g
                key={p.id}
                className={onGate ? "gate-marker clickable" : "gate-marker"}
                onClick={() => onGate?.(p.id)}
              >
                <circle
                  cx={x}
                  cy={y}
                  r="25"
                  fill={opened ? "#ffffff" : "#bd3449"}
                  stroke={changed ? "#df7637" : opened ? "#167552" : "#bd3449"}
                  strokeWidth="3"
                />
                <path
                  d={
                    opened
                      ? `M${x - 9} ${y}h18m-6-6 6 6-6 6`
                      : `m${x - 7} ${y - 7} 14 14m-14 0 14-14`
                  }
                  fill="none"
                  stroke={opened ? "#167552" : "#ffffff"}
                  strokeWidth="2.5"
                />
                <text
                  x={x < 70 ? x + 34 : x > 920 ? x - 34 : x}
                  y={x < 70 || x > 920 ? y + 5 : y + (y < 150 ? -34 : 43)}
                  textAnchor={x < 70 ? "start" : x > 920 ? "end" : "middle"}
                  className="map-gate-label"
                >
                  {p.name} · {opened ? "Open" : "Closed"}
                </text>
              </g>
            );
          })}
        <text x="72" y="932" className="map-meta">
          {venue.zones.length} ZONES /{" "}
          {venue.portals.filter((p) => p.kind === "exit").length} EXITS
        </text>
        <text x="920" y="932" textAnchor="end" className="map-meta">
          {counts
            ? "SIMULATED OCCUPANCY"
            : states
              ? "ESTIMATED / ASSUMED OCCUPANCY"
              : "AWAITING INPUT"}
        </text>
      </svg>
    </div>
  );
}
