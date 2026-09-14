"""Cached model inference and explainability helpers."""
import json
from pathlib import Path
from typing import Any, Mapping
import joblib
import pandas as pd
from .config import get_settings
from .opportunity import OpportunityEngine
from .risk import RiskEngine
from .schemas import FEATURES, validate_customer

class ModelPredictor:
    _cache = {}
    def __init__(self, model_path=None, metrics_path=None, risk_engine=None, opportunity_engine=None):
        settings = get_settings()
        self.model_path = Path(model_path or settings.model_path)
        self.metrics_path = Path(metrics_path or settings.metrics_path)
        key = str(self.model_path.resolve())
        if key not in self._cache:
            self._cache[key] = joblib.load(self.model_path)
        self.model = self._cache[key]
        self.risk_engine = risk_engine or RiskEngine()
        self.opportunity_engine = opportunity_engine or OpportunityEngine()
        self.metrics = json.loads(self.metrics_path.read_text(encoding="utf-8")) if self.metrics_path.exists() else {}
        self.threshold = float(self.metrics.get("decision_threshold", .5))
    def predict(self, customer: Mapping[str, Any]) -> dict:
        data = validate_customer(customer)
        probability = float(self.model.predict_proba(pd.DataFrame([data])[list(FEATURES)])[0, 1])
        risk = self.risk_engine.assess(probability, data)
        # Keep the predictor's configured opportunity weights authoritative.
        opportunity = self.opportunity_engine.score(data, probability, risk["level"])
        return {"probability": probability, "prediction": int(probability >= self.threshold),
                "threshold": self.threshold, "risk": risk, "opportunity": opportunity,
                "model": self.metrics.get("best_model", "")}
    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = [self.predict(row.to_dict()) for _, row in frame.iterrows()]
        output = frame.copy()
        output["Churn_Probability"] = [r["probability"] for r in rows]
        output["Risk_Score"] = [r["risk"]["score"] for r in rows]
        output["Risk_Level"] = [r["risk"]["level"] for r in rows]
        output["Recommended_Action"] = [r["risk"]["action"] for r in rows]
        output["Prediction"] = [r["prediction"] for r in rows]
        output["Opportunity_Score"] = [r["opportunity"]["score"] for r in rows]
        output["Customer_Value_Proxy"] = [r["opportunity"]["relationship_value_proxy"] for r in rows]
        output["Intervention_Opportunity"] = [r["opportunity"]["intervention_component"] for r in rows]
        output["Priority_Tier"] = [r["opportunity"]["priority_tier"] for r in rows]
        output["Opportunity_Segment"] = [r["opportunity"]["segment"] for r in rows]
        output["Recommended_Playbook"] = [r["opportunity"]["playbook"] for r in rows]
        return output.sort_values("Churn_Probability", ascending=False)
    def feature_importance(self) -> pd.DataFrame:
        try:
            prep, estimator = self.model.named_steps["preprocessor"], self.model.named_steps["model"]
            values = getattr(estimator, "feature_importances_", None)
            if values is None:
                values = abs(estimator.coef_[0])
            names = prep.get_feature_names_out()
            return pd.DataFrame({"Feature": [str(n).split("__", 1)[-1] for n in names],
                                 "Importance": values}).sort_values("Importance", ascending=False)
        except (AttributeError, KeyError, ValueError):
            return pd.DataFrame(columns=["Feature", "Importance"])
