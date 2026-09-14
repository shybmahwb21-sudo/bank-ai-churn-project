"""Bank integration abstraction and deterministic in-process demo client."""
from abc import ABC, abstractmethod
from copy import deepcopy
from uuid import uuid4
from .schemas import validate_customer

class BankClient(ABC):
    @abstractmethod
    def list_customers(self): ...
    @abstractmethod
    def get_customer(self, customer_id): ...
    @abstractmethod
    def create_customer(self, payload): ...
    @abstractmethod
    def sync(self): ...

class OfficialBankClient(BankClient):
    """Explicit adapter seam for a future documented bank API.

    It intentionally fails closed until an official contract and authenticated
    transport are supplied; Demo Mode never uses this class.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Official bank integration is not configured. "
            "Use BANK_API_MODE=mock until an approved API adapter is implemented."
        )

class MockBankClient(BankClient):
    def __init__(self, customers=None):
        self.customers = {str(c.get("customer_id", c.get("CustomerId", uuid4().hex))): deepcopy(c)
                          for c in (customers or self._demo_customers())}
    @staticmethod
    def _demo_customers():
        # Deterministic synthetic portfolio for a convincing academic demo.
        # These records are not copied from the training dataset or a real bank.
        profiles = [
            ("DEMO-001", 620, "Germany", "Female", 45, 3, 90000, 1, 1, 0, 85000),
            ("DEMO-002", 750, "France", "Male", 32, 8, 50000, 2, 1, 1, 120000),
            ("DEMO-003", 590, "Germany", "Male", 51, 2, 125000, 1, 1, 0, 64000),
            ("DEMO-004", 680, "Spain", "Female", 39, 6, 72000, 2, 1, 1, 98000),
            ("DEMO-005", 710, "France", "Female", 29, 4, 0, 1, 0, 1, 54000),
            ("DEMO-006", 645, "Germany", "Female", 47, 9, 148000, 3, 1, 0, 76000),
            ("DEMO-007", 790, "Spain", "Male", 35, 10, 42000, 2, 1, 1, 132000),
            ("DEMO-008", 605, "France", "Male", 58, 1, 98000, 1, 0, 0, 47000),
            ("DEMO-009", 735, "Germany", "Female", 41, 7, 86000, 2, 1, 1, 110000),
            ("DEMO-010", 665, "Spain", "Female", 52, 5, 30000, 1, 1, 0, 69000),
            ("DEMO-011", 770, "France", "Male", 26, 3, 64000, 2, 1, 1, 145000),
            ("DEMO-012", 625, "Germany", "Male", 44, 0, 112000, 1, 1, 0, 58000),
        ]
        fields = ("customer_id", "CreditScore", "Geography", "Gender", "Age", "Tenure",
                  "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary")
        return [dict(zip(fields, profile)) for profile in profiles]
    def list_customers(self): return deepcopy(list(self.customers.values()))
    def get_customer(self, customer_id):
        return deepcopy(self.customers.get(str(customer_id)))
    def create_customer(self, payload):
        data = validate_customer(payload)
        cid = str(payload.get("customer_id", uuid4().hex))
        data["customer_id"] = cid
        self.customers[cid] = deepcopy(data)
        return deepcopy(data)
    def sync(self): return {"customers": self.list_customers(), "count": len(self.customers)}
    def health(self): return {"status": "ok", "mode": "demo"}
    def statistics(self): return {"customers": len(self.customers)}
    def webhook(self, event): return {"accepted": True, "event_id": event.get("id", uuid4().hex)}
