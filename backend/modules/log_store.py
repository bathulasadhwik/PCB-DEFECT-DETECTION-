import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class LocalLogStore:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.enabled = True
        self.error: Optional[str] = None

        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_scan(self, payload: Dict) -> Dict:
        record = {
            "timestamp": payload.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
            "source": payload.get("source", "upload"),
            "pcb_name": payload.get("pcb_name", ""),
            "status": payload.get("status", "UNKNOWN"),
            "risk_level": payload.get("risk_level", "UNKNOWN"),
            "failure_probability": str(payload.get("failure_probability", "N/A")),
            "defects": payload.get("defects", []),
            "report_path": payload.get("report_path", ""),
        }
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            return {"success": True}
        except Exception as exc:
            self.error = str(exc)
            return {"success": False, "error": f"Local log write failed: {exc}"}

    def fetch_logs(self, limit: int = 200) -> List[Dict]:
        if not os.path.exists(self.log_file_path):
            return []

        rows: List[Dict] = []
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
        except Exception as exc:
            self.error = str(exc)
            return []

        rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return rows[:limit]
