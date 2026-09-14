"""Configurable 0-100 risk bands and editable recommended actions."""
from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional
from .opportunity import OpportunityEngine

@dataclass(frozen=True)
class RiskBand:
    name: str
    minimum: int
    maximum: int
    action: str

DEFAULT_BANDS = (
    RiskBand("LOW", 0, 24, "Routine follow-up"),
    RiskBand("MEDIUM", 25, 49, "Contact and support"),
    RiskBand("HIGH", 50, 74, "Personalized retention offer"),
    RiskBand("CRITICAL", 75, 100, "Immediate specialist review"),
)

class RiskEngine:
    def __init__(self, bands=DEFAULT_BANDS, actions: Optional[Dict[str, str]] = None,
                 opportunity_engine=None):
        self.bands = tuple(bands)
        self.actions = dict(actions or {})
        self.opportunity_engine = opportunity_engine or OpportunityEngine()
        self._validate()
    def _validate(self):
        if len(self.bands) != 4 or self.bands[0].minimum != 0 or self.bands[-1].maximum != 100:
            raise ValueError("Risk bands must cover 0-100")
        for previous, current in zip(self.bands, self.bands[1:]):
            if previous.maximum + 1 != current.minimum:
                raise ValueError("Risk bands must be contiguous")
        if any(b.minimum < 0 or b.maximum > 100 or b.minimum > b.maximum for b in self.bands):
            raise ValueError("Invalid risk band")
    def assess(self, probability: float, customer: Optional[Mapping[str, Any]] = None):
        score = max(0.0, min(100.0, float(probability) * 100))
        if not math.isfinite(score):
            raise ValueError("Probability must be a finite number")
        # Bands are defined over integer score points; flooring avoids a gap
        # between 74 and 75 when a model returns a fractional score.
        band_score = min(100, math.floor(score))
        band = next((b for b in self.bands if b.minimum <= band_score <= b.maximum), None)
        if band is None:
            raise ValueError(f"No risk band covers score {score:.6f}")
        result = {"score": score, "level": band.name, "action": self.actions.get(band.name, band.action)}
        if customer is not None:
            result["opportunity"] = self.opportunity_engine.score(customer, probability, band.name)
        return result
    def set_action(self, level: str, action: str):
        if level not in {b.name for b in self.bands} or not action.strip():
            raise ValueError("Unknown level or empty action")
        self.actions[level] = action.strip()
