import { useId } from "react";
import { number } from "./api";
export function Chart({
  series,
  labels,
  height = 160,
}: {
  series: { name: string; values: number[]; color: string }[];
  labels?: string[];
  height?: number;
}) {
  const id = useId().replace(/:/g, "");
  const max = Math.max(1, ...series.flatMap((s) => s.values)) * 1.12;
  const width = 720;
  const pad = 38;
  const chartH = height - 32;
  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={series
          .map((s) => `${s.name}: ${s.values.length} data points`)
          .join("; ")}
      >
        <defs>
          <linearGradient id={`fill-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0"
              stopColor={series[0]?.color || "#2456d6"}
              stopOpacity=".16"
            />
            <stop
              offset="1"
              stopColor={series[0]?.color || "#2456d6"}
              stopOpacity="0"
            />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((n) => (
          <g key={n}>
            <line
              x1={pad}
              x2={width - 8}
              y1={8 + n * (chartH - 8)}
              y2={8 + n * (chartH - 8)}
              stroke="#e3e9f1"
              strokeDasharray="4 4"
            />
            <text
              x={pad - 8}
              y={12 + n * (chartH - 8)}
              textAnchor="end"
              className="chart-label"
            >
              {number(max * (1 - n))}
            </text>
          </g>
        ))}
        {series.map((s, i) => {
          const points = s.values.map(
            (v, j) =>
              `${pad + (j * (width - pad - 8)) / Math.max(1, s.values.length - 1)},${chartH - (v / max) * (chartH - 8)}`,
          );
          return (
            <g key={s.name}>
              {i === 0 && points.length > 1 && (
                <path
                  d={`M${pad},${chartH} L${points.join(" L")} L${width - 8},${chartH}Z`}
                  fill={`url(#fill-${id})`}
                />
              )}
              <polyline
                points={points.join(" ")}
                fill="none"
                stroke={s.color}
                strokeWidth="2.5"
                strokeLinejoin="round"
              />
              {points.length === 1 && (
                <circle
                  cx={pad}
                  cy={chartH - (s.values[0] / max) * (chartH - 8)}
                  r="4"
                  fill={s.color}
                />
              )}
            </g>
          );
        })}
        {labels?.map((l, i) => (
          <text
            key={i}
            x={pad + (i * (width - pad - 8)) / Math.max(1, labels.length - 1)}
            y={height - 5}
            textAnchor={
              i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle"
            }
            className="chart-label"
          >
            {l}
          </text>
        ))}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}
