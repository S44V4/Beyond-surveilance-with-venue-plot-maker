from __future__ import annotations

from typing import Any


def _class_index(zone: dict[str, Any]) -> int:
    return int(zone.get("risk_class_index", 0))


def _alert_class_index(zone: dict[str, Any]) -> int:
    return max(1, int(zone.get("risk_class_count", 4)) // 2)


def build_model_alerts(
    zones: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    global_summary: dict[str, Any],
    *,
    maximum_current_alerts: int = 3,
) -> list[dict[str, Any]]:
    """Turn learned risk outputs into a small, auditable operator alert feed.

    This is deliberately a presentation rule over model evidence, not another
    classifier. Alerts follow the learned high/critical ordinal class directly;
    no invented numeric probability threshold is applied.
    """

    ranked = sorted(
        (
            zone
            for zone in zones
            if _class_index(zone) >= _alert_class_index(zone)
        ),
        key=lambda zone: (
            _class_index(zone),
            float(zone.get("risk", 0.0)),
            float(zone.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    alerts: list[dict[str, Any]] = []
    for zone in ranked[:maximum_current_alerts]:
        level = str(zone.get("risk_level", "high"))
        severity = (
            "critical"
            if _class_index(zone) == int(zone.get("risk_class_count", 4)) - 1
            else "warning"
        )
        zone_id = str(zone.get("id", "unknown"))
        risk = float(zone.get("risk", 0.0))
        confidence = float(zone.get("confidence", 0.0))
        alerts.append(
            {
                "id": f"current-risk-{zone_id}",
                "kind": "crowd_abnormality",
                "severity": severity,
                "title": (
                    "Critical crowd-pressure pattern detected"
                    if severity == "critical"
                    else "High crowd-pressure pattern detected"
                ),
                "message": (
                    f"{zone.get('name', zone_id)} is classified in the {level} "
                    f"pressure-proxy class ({risk:.1f} ordinal score, "
                    f"{confidence:.1f}% confidence)."
                ),
                "zone_id": zone_id,
                "risk": risk,
                "confidence": confidence,
                "horizon_seconds": None,
                "source": "graph_pressure_proxy_head",
                "operator_action": "Inspect the highlighted zone and verify the camera feed.",
            }
        )

    if forecasts:
        forecast_candidates = [
            (forecast, zone)
            for forecast in forecasts
            for zone in forecast.get("zones", [])
            if _class_index(zone) >= _alert_class_index(zone)
        ]
        if forecast_candidates:
            peak, peak_zone = max(
                forecast_candidates,
                key=lambda item: (
                    _class_index(item[1]),
                    float(item[1].get("risk", 0.0)),
                ),
            )
            peak_risk = float(peak_zone.get("risk", 0.0))
            horizon = int(peak.get("horizon_seconds", 0))
            zone_id = str(peak_zone.get("id", "unknown"))
            alerts.append(
                {
                    "id": f"forecast-risk-{horizon}-{zone_id}",
                    "kind": "forecast_abnormality",
                    "severity": "warning",
                    "title": "Future high crowd-pressure class predicted",
                    "message": (
                        f"{zone_id} is forecast in the {peak_zone.get('risk_level')} "
                        f"pressure class at step {horizon}."
                    ),
                    "zone_id": zone_id,
                    "risk": peak_risk,
                    "confidence": None,
                    "horizon_seconds": horizon,
                    "source": "graph_pressure_proxy_forecast_head",
                    "operator_action": "Monitor the highlighted zone and prepare crowd-control staff.",
                }
            )
    return alerts
