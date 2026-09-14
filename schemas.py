"""Validated, serialisable contracts shared by UI, clients and persistence."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

FEATURES = ("CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary")

class ValidationError(ValueError):
    """Raised when an integration payload is unsafe or incomplete."""

def validate_customer(payload: Mapping[str, Any], partial: bool = False) -> Dict[str, Any]:
    data = dict(payload)
    missing = [] if partial else [f for f in FEATURES if f not in data]
    if missing:
        raise ValidationError("Missing customer fields: " + ", ".join(missing))
    numeric = {"CreditScore": (0, 1000), "Age": (0, 130), "Tenure": (0, 100),
               "Balance": (0, None), "NumOfProducts": (1, 20),
               "HasCrCard": (0, 1), "IsActiveMember": (0, 1),
               "EstimatedSalary": (0, None)}
    for key, (low, high) in numeric.items():
        if key not in data:
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{key} must be numeric") from exc
        if value < low or (high is not None and value > high):
            raise ValidationError(f"{key} is outside the allowed range")
    if "Geography" in data and not str(data["Geography"]).strip():
        raise ValidationError("Geography cannot be empty")
    if "Gender" in data and not str(data["Gender"]).strip():
        raise ValidationError("Gender cannot be empty")
    return data

@dataclass
class PredictionRecord:
    customer_id: str
    probability: float
    risk_level: str
    action: str
    model_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class SyncResult:
    sync_id: str
    fetched: int
    created: int
    updated: int
    failed: int
    duration_ms: float
    errors: list = field(default_factory=list)
