"""Audit facade that never logs secrets or raw credentials."""
class AuditService:
    REDACT = {"password", "token", "secret", "authorization", "api_key"}
    def __init__(self, store): self.store = store
    def record(self, event, detail=None):
        detail = {k: ("[REDACTED]" if k.lower() in self.REDACT else v)
                  for k, v in (detail or {}).items()}
        self.store.audit(event, detail)
    def recent(self, limit=100): return self.store.list_audit(limit)
