"""Operational health and safe service diagnostics."""
from time import perf_counter
class HealthService:
    def __init__(self, store, client): self.store, self.client = store, client
    def check(self):
        start = perf_counter()
        try:
            self.client.health() if hasattr(self.client, "health") else True
            self.store.conn.execute("SELECT 1")
            return {"status": "ok", "latency_ms": round((perf_counter()-start)*1000, 2), "database": "ok"}
        except Exception as exc:
            return {"status": "degraded", "latency_ms": round((perf_counter()-start)*1000, 2),
                    "database": "error", "error": str(exc)}
