import tempfile
import unittest
from pathlib import Path

from src.audit import AuditService
from src.bank_client import MockBankClient
from src.persistence import SQLiteStore
from src.opportunity import (
    OpportunityConfig,
    calculate_opportunity_score,
    explain_opportunity,
    simulate_campaign,
)
from src.risk import RiskEngine
from src.schemas import ValidationError, validate_customer
from src.sync_service import SyncService

CUSTOMER = {"CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 40,
            "Tenure": 5, "Balance": 1000, "NumOfProducts": 1, "HasCrCard": 1,
            "IsActiveMember": 1, "EstimatedSalary": 50000}

class BankAITest(unittest.TestCase):
    def test_validation_and_risk_bands(self):
        self.assertEqual(validate_customer(CUSTOMER)["Age"], 40)
        with self.assertRaises(ValidationError): validate_customer({"Age": "bad"})
        engine = RiskEngine()
        self.assertEqual(engine.assess(.10)["level"], "LOW")
        self.assertEqual(engine.assess(.25)["level"], "MEDIUM")
        self.assertEqual(engine.assess(.75)["level"], "CRITICAL")
        engine.set_action("HIGH", "Call manager")
        self.assertEqual(engine.assess(.60)["action"], "Call manager")

    def test_mock_client_and_sync_audit(self):
        client = MockBankClient()
        created = client.create_customer(CUSTOMER)
        self.assertEqual(client.get_customer(created["customer_id"])["Age"], 40)
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteStore(Path(folder) / "test.sqlite")
            result = SyncService(client, store).run()
            self.assertGreaterEqual(result.created, 2)
            audit = AuditService(store)
            audit.record("test", {"token": "secret", "safe": "yes"})
            self.assertEqual(audit.recent(1)[0]["event"], "test")
            self.assertIn("[REDACTED]", audit.recent(1)[0]["detail"])
            store.close()

    def test_opportunity_score_is_explainable_and_configurable(self):
        customer = dict(CUSTOMER, Balance=80_000, IsActiveMember=0, NumOfProducts=1)
        explanation = explain_opportunity(customer, 0.80, "CRITICAL")
        self.assertGreater(explanation["score"], 50)
        self.assertEqual(explanation["priority_tier"], "P1 · Act now")
        self.assertIn("risk", explanation["explanation"])
        self.assertGreater(explanation["intervention_component"], 0)
        risk_only = OpportunityConfig(risk_weight=1, value_weight=0, intervention_weight=0)
        self.assertAlmostEqual(calculate_opportunity_score(customer, .4, risk_only), 40.0)

    def test_campaign_simulator_uses_demo_scenario_arithmetic(self):
        rows = [
            {"Churn_Probability": .5, "Customer_Value_Proxy": 1000},
            {"Churn_Probability": .2, "Customer_Value_Proxy": 500},
        ]
        result = simulate_campaign(rows, intervention_cost=20, expected_save_rate=.25)
        self.assertEqual(result["targeted_customers"], 2)
        self.assertEqual(result["baseline_proxy_value_at_risk"], 600.0)
        self.assertEqual(result["scenario_estimated_value_preserved"], 150.0)
        self.assertEqual(result["scenario_total_cost"], 40.0)
        with self.assertRaises(ValueError):
            simulate_campaign(rows, intervention_cost=20, expected_save_rate=1.1)

if __name__ == "__main__":
    unittest.main()
