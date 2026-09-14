"""Idempotent bank-to-local synchronization with timing and audit records."""
from time import perf_counter
from uuid import uuid4
from .schemas import SyncResult

class SyncService:
    def __init__(self, client, store, predictor=None):
        self.client, self.store, self.predictor = client, store, predictor
    def run(self):
        started, sync_id = perf_counter(), uuid4().hex
        created = updated = failed = 0; errors = []
        for customer in self.client.list_customers():
            try:
                old = self.store.conn.execute("SELECT 1 FROM customers WHERE customer_id=?",
                                               (str(customer["customer_id"]),)).fetchone()
                self.store.upsert_customer(customer)
                updated += bool(old); created += not bool(old)
                if self.predictor:
                    prediction = self.predictor.predict(customer)
                    self.store.save_prediction(customer["customer_id"], prediction)
            except Exception as exc:
                failed += 1; errors.append(str(exc))
        result = SyncResult(sync_id, created + updated + failed, created, updated, failed,
                            (perf_counter() - started) * 1000, errors)
        self.store.save_sync(sync_id, result.__dict__); self.store.audit("sync.completed", result.__dict__)
        return result
