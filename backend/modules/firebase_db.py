import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import firebase_admin  # type: ignore[reportMissingImports]
    from firebase_admin import credentials, db  # type: ignore[reportMissingImports]
except Exception:
    firebase_admin = None
    credentials = None
    db = None


class FirebaseLogger:
    _initialized = False

    def __init__(self):
        self.enabled = False
        self.error: Optional[str] = None
        self.base_ref: Optional[Any] = None

        if firebase_admin is None or credentials is None or db is None:
            self.error = "Firebase SDK missing: install dependencies from requirements.txt"
            return

        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip()
        cred_json_raw = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
        db_url = os.getenv("FIREBASE_DB_URL", "").strip()

        if (not cred_path and not cred_json_raw) or not db_url:
            self.error = "Firebase disabled: set FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON, and FIREBASE_DB_URL"
            return

        try:
            if not FirebaseLogger._initialized:
                if cred_json_raw:
                    cred_info = json.loads(cred_json_raw)
                    cred = credentials.Certificate(cred_info)
                else:
                    cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {"databaseURL": db_url})
                FirebaseLogger._initialized = True

            self.base_ref = db.reference("pcb_logs")
            self.enabled = True
        except Exception as exc:
            self.error = f"Firebase init failed: {exc}"
            self.enabled = False

    def log_scan(self, payload: Dict) -> Dict:
        if not self.enabled or self.base_ref is None:
            return {"success": False, "error": self.error or "Firebase disabled"}

        try:
            ref = self.base_ref
            record = {
                "timestamp": payload.get("timestamp", datetime.utcnow().isoformat()),
                "source": payload.get("source", "upload"),
                "status": payload.get("status", "UNKNOWN"),
                "risk_level": payload.get("risk_level", "UNKNOWN"),
                "failure_probability": str(payload.get("failure_probability", "N/A")),
                "defects": payload.get("defects", []),
                "report_path": payload.get("report_path", ""),
            }
            ref.push(record)
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": f"Firebase write failed: {exc}"}

    def fetch_logs(self, limit: int = 200) -> List[Dict]:
        if not self.enabled or self.base_ref is None:
            return []

        try:
            ref = self.base_ref
            data = ref.order_by_key().limit_to_last(limit).get()
            if not data:
                return []

            logs = []
            for key, value in data.items():
                if isinstance(value, dict):
                    item = {"id": key, **value}
                    logs.append(item)

            logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return logs
        except Exception:
            return []
