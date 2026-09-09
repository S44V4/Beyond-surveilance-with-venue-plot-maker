from __future__ import annotations

from typing import Any, Protocol


class ExplanationProvider(Protocol):
    name: str

    def explain(self, evidence: dict[str, Any]) -> str: ...


class DeterministicEvidenceExplainer:
    """Auditable fallback summary; explicitly not presented as an LLM."""

    name = "deterministic_evidence"

    def explain(self, evidence: dict[str, Any]) -> str:
        global_state = evidence.get("global", {})
        zones = sorted(
            evidence.get("zones", []), key=lambda zone: zone.get("risk", 0), reverse=True
        )
        highest = zones[0] if zones else None
        forecasts = evidence.get("forecasts", [])
        peak = max(forecasts, key=lambda item: item.get("max_risk", 0), default=None)
        statements = [
            (
                f"The model estimates {global_state.get('estimated_count', 0):.0f} people "
                f"with a maximum current risk score of "
                f"{global_state.get('max_risk', 0):.1f}%."
            )
        ]
        if highest:
            statements.append(
                f"{highest.get('id')} is the highest-risk zone at "
                f"{highest.get('risk', 0):.1f}% and estimated occupancy "
                f"{highest.get('count', 0):.0f}."
            )
        if peak:
            statements.append(
                f"The largest forecast risk is {peak.get('max_risk', 0):.1f}% at "
                f"+{peak.get('horizon_seconds', 0)} seconds."
            )
        statements.append(
            "This summary describes model evidence only; an operator must approve any action."
        )
        return " ".join(statements)


class CrowdIntelligenceService:
    """Keeps explanation providers separate from safety-critical decisions."""

    def __init__(self, provider: ExplanationProvider | None = None) -> None:
        self.provider = provider or DeterministicEvidenceExplainer()

    def generate(self, evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": self.provider.name,
            "is_decision_maker": False,
            "explanation": self.provider.explain(evidence),
        }
