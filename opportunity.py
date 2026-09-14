"""Explainable retention opportunity scoring for the academic demo.

The opportunity engine is deliberately separate from the churn model. It never
changes model probabilities; it combines the probability with transparent,
configurable demo proxies so a team can decide where an intervention may be
most useful.
"""
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class OpportunityConfig:
    """Weights and scale used by the opportunity score.

    ``value_reference`` is a display calibration for the demo relationship
    value proxy, not a currency valuation or a bank financial assumption.
    """

    risk_weight: float = 0.45
    value_weight: float = 0.35
    intervention_weight: float = 0.20
    value_reference: float = 100_000.0

    def __post_init__(self):
        weights = (self.risk_weight, self.value_weight, self.intervention_weight)
        if any(weight < 0 or weight > 1 for weight in weights):
            raise ValueError("Opportunity weights must be between 0 and 1")
        if sum(weights) <= 0:
            raise ValueError("At least one opportunity weight must be positive")
        if self.value_reference <= 0:
            raise ValueError("value_reference must be positive")


DEFAULT_OPPORTUNITY_CONFIG = OpportunityConfig()


def _number(customer: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(customer.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(value, 0.0)


def relationship_value_proxy(customer: Mapping[str, Any]) -> float:
    """Return a clearly non-financial demo proxy for relationship value."""
    return _number(customer, "Balance") + (0.10 * _number(customer, "EstimatedSalary"))


def intervention_opportunity(customer: Mapping[str, Any]) -> float:
    """Estimate how many transparent intervention signals are present."""
    score = 0.20
    if not bool(customer.get("IsActiveMember", 1)):
        score += 0.35
    if _number(customer, "NumOfProducts", 1) <= 1:
        score += 0.20
    if 0 < _number(customer, "Balance") <= 100_000:
        score += 0.10
    if _number(customer, "Tenure") <= 3:
        score += 0.15
    return min(score, 1.0)


def opportunity_segment(customer: Mapping[str, Any]) -> str:
    """Create an actionable, non-sensitive demo segment label."""
    geography = str(customer.get("Geography", "Other")).strip() or "Other"
    activity = "Inactive relationship" if not bool(customer.get("IsActiveMember", 1)) else "Active relationship"
    products = "Single product" if _number(customer, "NumOfProducts", 1) <= 1 else "Multi-product"
    return f"{geography} · {activity} · {products}"


def priority_tier(score: float) -> str:
    """Map an opportunity score to a simple operating tier."""
    value = float(score)
    if value >= 75:
        return "P1 · Act now"
    if value >= 50:
        return "P2 · Next"
    if value >= 25:
        return "P3 · Plan"
    return "P4 · Monitor"


def recommended_playbook(risk_level: str, customer: Mapping[str, Any]) -> str:
    """Return a risk/segment-aware starting playbook, not an automatic decision."""
    level = str(risk_level or "").upper()
    inactive = not bool(customer.get("IsActiveMember", 1))
    single_product = _number(customer, "NumOfProducts", 1) <= 1
    if level == "CRITICAL":
        return "Specialist outreach + service recovery" if inactive else "Priority advisor call + tailored save plan"
    if level == "HIGH":
        return "Reactivation check-in + product-fit review" if inactive else (
            "Advisor call + relationship deepening" if single_product else "Advisor call + value review"
        )
    if level == "MEDIUM":
        return "Proactive check-in + needs discovery" if single_product else "Digital nurture + needs discovery"
    return "Monitor and nurture; no urgent outreach"


def explain_opportunity(
    customer: Mapping[str, Any],
    churn_probability: float,
    risk_level: str = "",
    config: Optional[OpportunityConfig] = None,
) -> dict:
    """Return score, components, tier, and plain-language explanation."""
    config = config or DEFAULT_OPPORTUNITY_CONFIG
    probability = min(max(float(churn_probability), 0.0), 1.0)
    value_proxy = relationship_value_proxy(customer)
    value_component = min(value_proxy / config.value_reference, 1.0)
    intervention_component = intervention_opportunity(customer)
    total_weight = config.risk_weight + config.value_weight + config.intervention_weight
    score = 100.0 * (
        config.risk_weight * probability
        + config.value_weight * value_component
        + config.intervention_weight * intervention_component
    ) / total_weight
    tier = priority_tier(score)
    level = risk_level or ("CRITICAL" if probability >= .75 else "HIGH" if probability >= .60
                           else "MEDIUM" if probability >= .30 else "LOW")
    return {
        "score": round(score, 2),
        "risk_component": round(probability, 4),
        "value_component": round(value_component, 4),
        "intervention_component": round(intervention_component, 4),
        "relationship_value_proxy": round(value_proxy, 2),
        "priority_tier": tier,
        "segment": opportunity_segment(customer),
        "playbook": recommended_playbook(level, customer),
        "explanation": (
            f"{config.risk_weight:.0%} risk × {probability:.1%}, "
            f"{config.value_weight:.0%} demo value × {value_component:.1%}, "
            f"{config.intervention_weight:.0%} intervention opportunity × "
            f"{intervention_component:.1%}"
        ),
    }


def calculate_opportunity_score(
    customer: Mapping[str, Any],
    churn_probability: float,
    config: Optional[OpportunityConfig] = None,
) -> float:
    """Convenience function returning only the 0–100 score."""
    return float(explain_opportunity(customer, churn_probability, config=config)["score"])


class OpportunityEngine:
    """Small service wrapper suitable for predictor and API integrations."""

    def __init__(self, config: Optional[OpportunityConfig] = None):
        self.config = config or DEFAULT_OPPORTUNITY_CONFIG

    def score(self, customer: Mapping[str, Any], churn_probability: float, risk_level: str = "") -> dict:
        return explain_opportunity(customer, churn_probability, risk_level, self.config)


def simulate_campaign(
    records: Iterable[Mapping[str, Any]],
    intervention_cost: float,
    expected_save_rate: float,
) -> dict:
    """Calculate scenario arithmetic for selected records.

    Outputs are explicitly estimates based on demo proxies. They are not
    observed saves, forecasts, ROI claims, or financial advice.
    """
    cost = float(intervention_cost)
    save_rate = float(expected_save_rate)
    if cost < 0:
        raise ValueError("intervention_cost must be non-negative")
    if not 0 <= save_rate <= 1:
        raise ValueError("expected_save_rate must be between 0 and 1")
    rows = records.to_dict("records") if hasattr(records, "to_dict") else list(records)
    customers = len(rows)
    baseline = sum(
        min(max(float(row.get("Churn_Probability", 0.0)), 0.0), 1.0)
        * max(float(row.get("Customer_Value_Proxy", row.get("relationship_value_proxy", 0.0))), 0.0)
        for row in rows
    )
    estimated_saved = customers * save_rate
    estimated_preserved = baseline * save_rate
    total_cost = customers * cost
    net_proxy = estimated_preserved - total_cost
    roi_proxy = estimated_preserved / total_cost if total_cost else None
    return {
        "targeted_customers": customers,
        "intervention_cost": cost,
        "expected_save_rate": save_rate,
        "baseline_proxy_value_at_risk": round(baseline, 2),
        "scenario_estimated_saved_customers": round(estimated_saved, 2),
        "scenario_estimated_value_preserved": round(estimated_preserved, 2),
        "scenario_total_cost": round(total_cost, 2),
        "scenario_net_proxy_value": round(net_proxy, 2),
        "scenario_roi_proxy": round(roi_proxy, 2) if roi_proxy is not None else None,
        "is_demo_scenario": True,
    }


__all__ = [
    "OpportunityConfig", "DEFAULT_OPPORTUNITY_CONFIG", "OpportunityEngine",
    "calculate_opportunity_score", "explain_opportunity", "simulate_campaign",
    "relationship_value_proxy", "intervention_opportunity", "priority_tier",
    "recommended_playbook", "opportunity_segment",
]
