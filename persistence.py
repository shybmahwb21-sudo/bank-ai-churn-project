"""Small SQLite repository for operational records."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

class SQLiteStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""CREATE TABLE IF NOT EXISTS customers
            (customer_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions
            (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sync_runs
            (sync_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_log
            (id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, detail TEXT, created_at TEXT NOT NULL);""")
        self.conn.commit()
    @staticmethod
    def _now(): return datetime.now(timezone.utc).isoformat()
    def upsert_customer(self, customer):
        cid = str(customer["customer_id"])
        self.conn.execute("INSERT OR REPLACE INTO customers VALUES (?,?,?)", (cid, json.dumps(customer), self._now()))
        self.conn.commit()
    def save_prediction(self, customer_id, prediction):
        self.conn.execute("INSERT INTO predictions(customer_id,payload,created_at) VALUES (?,?,?)",
                          (str(customer_id), json.dumps(prediction), self._now())); self.conn.commit()
    def save_sync(self, sync_id, result):
        self.conn.execute("INSERT OR REPLACE INTO sync_runs VALUES (?,?,?)",
                          (sync_id, json.dumps(result), self._now())); self.conn.commit()
    def audit(self, event, detail=None):
        self.conn.execute("INSERT INTO audit_log(event,detail,created_at) VALUES (?,?,?)",
                          (event, json.dumps(detail or {}), self._now())); self.conn.commit()
    def list_audit(self, limit=100):
        return [dict(row) for row in self.conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]
    def close(self):
        self.conn.close()
