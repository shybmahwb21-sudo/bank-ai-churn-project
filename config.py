"""Environment-backed configuration with safe demo defaults."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))

@dataclass(frozen=True)
class Settings:
    demo_mode: bool = True
    bank_api_mode: str = "mock"
    database_path: Path = ROOT / "outputs" / "bank_ai.sqlite3"
    model_path: Path = ROOT / "best_model.joblib"
    metrics_path: Path = ROOT / "outputs" / "metrics.json"
    webhook_secret: str = ""
    api_url: str = ""
    api_key: str = ""
    api_timeout: float = 5.0

def get_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    return Settings(
        demo_mode=os.getenv("DEMO_MODE", "true").lower() in {"1", "true", "yes"},
        bank_api_mode=os.getenv("BANK_API_MODE", "mock").lower(),
        database_path=Path(os.getenv("DATABASE_PATH", str(ROOT / "outputs" / "bank_ai.sqlite3"))),
        model_path=Path(os.getenv("MODEL_PATH", str(ROOT / "best_model.joblib"))),
        metrics_path=Path(os.getenv("METRICS_PATH", str(ROOT / "outputs" / "metrics.json"))),
        webhook_secret=os.getenv("WEBHOOK_SECRET", ""),
        api_url=os.getenv("BANK_API_URL", ""),
        api_key=os.getenv("BANK_API_KEY", ""),
        api_timeout=float(os.getenv("BANK_API_TIMEOUT", "5")),
    )
