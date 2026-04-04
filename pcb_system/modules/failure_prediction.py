from collections import Counter
from typing import Dict, List


SEVERITY_WEIGHTS = {
    "missing": 0.85,
    "short": 0.95,
    "open circuit": 0.90,
    "shifted": 0.70,
    "misaligned": 0.70,
    "solder_bridge": 0.92,
    "scratch": 0.45,
    "default": 0.55,
}


def _get_weight(label: str) -> float:
    label_l = (label or "").strip().lower()
    for key, val in SEVERITY_WEIGHTS.items():
        if key != "default" and key in label_l:
            return val
    return SEVERITY_WEIGHTS["default"]


def predict_failure(defects: List[Dict]) -> Dict:
    if not defects:
        return {
            "failure_probability": 2.0,
            "risk_level": "LOW",
            "defect_distribution": {},
            "severity_score": 0.0,
        }

    weighted_scores = []
    labels = []
    for defect in defects:
        label = defect.get("label", "unknown")
        conf = float(defect.get("confidence", 0.5))
        weight = _get_weight(label)
        weighted_scores.append(conf * weight)
        labels.append(label)

    severity_score = min(1.0, sum(weighted_scores) / max(1, len(defects)) + min(0.25, 0.03 * len(defects)))
    failure_probability = round(severity_score * 100, 2)

    if failure_probability < 30:
        risk = "LOW"
    elif failure_probability < 70:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    distribution = dict(Counter(labels))

    return {
        "failure_probability": failure_probability,
        "risk_level": risk,
        "defect_distribution": distribution,
        "severity_score": round(severity_score, 4),
    }
