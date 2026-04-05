import base64
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from dotenv import load_dotenv

from modules.detection import YOLODetector, DetectionError
from modules.failure_prediction import predict_failure
from modules.log_store import LocalLogStore
from modules.gemini_ai import GeminiEngine
from modules.heatmap import generate_heatmap_overlay, save_heatmap
from modules.repair_engine import suggest_repairs
from modules.report_generator import generate_pdf_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)


def load_environment():
    env_candidates = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(WORKSPACE_DIR, ".env"),
    ]
    for env_path in env_candidates:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)


load_environment()

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
DETECTIONS_DIR = os.path.join(BASE_DIR, "detections")
HEATMAPS_DIR = os.path.join(BASE_DIR, "heatmaps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
RUNTIME_DIRS = [CAPTURES_DIR, DETECTIONS_DIR, HEATMAPS_DIR, REPORTS_DIR]

for d in RUNTIME_DIRS:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")


detector = None
model_error = None
try:
    detector = YOLODetector()
except Exception as exc:
    model_error = str(exc)

gemini_engine = GeminiEngine()
log_store = LocalLogStore(os.path.join(BASE_DIR, "logs", "scan_logs.jsonl"))

camera_thread = None
camera_running = False
camera_lock = threading.Lock()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_runtime_dirs():
    for d in RUNTIME_DIRS:
        os.makedirs(d, exist_ok=True)


def write_image_or_raise(path: str, image: np.ndarray, label: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, image)
    if not ok or not os.path.exists(path):
        raise IOError(f"Failed to write {label} image: {path}")


def encode_image_to_base64(image_path: str) -> str:
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def encode_image_to_base64_safe(image_path: str, fallback_data_url: str = "") -> str:
    if image_path and os.path.exists(image_path):
        return encode_image_to_base64(image_path)
    return fallback_data_url or ""


def encode_ndarray_to_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def decode_data_url_to_image(data_url: str):
    if not data_url or "," not in data_url:
        raise ValueError("Invalid data URL")
    _, b64_data = data_url.split(",", 1)
    raw = base64.b64decode(b64_data)
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode frame image")
    return image


def draw_annotated_detections(frame: np.ndarray, defects: List[Dict]) -> np.ndarray:
    annotated = frame.copy()
    for defect in defects:
        bbox = defect.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = str(defect.get("label", "unknown"))
        model_conf = float(defect.get("confidence", 0.0))
        fused_conf = float(defect.get("confidence_fused", model_conf))
        ai_verdict = str(defect.get("ai_verdict", "")).strip().upper()

        if ai_verdict == "CONFIRMED":
            color = (0, 200, 0)
        elif ai_verdict == "SUSPECT_FALSE_POSITIVE":
            color = (0, 0, 255)
        else:
            color = (0, 140, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        title = f"{label} m:{model_conf:.2f} a:{fused_conf:.2f}"
        if ai_verdict and ai_verdict != "UNCERTAIN":
            title = f"{title} [{ai_verdict}]"
        cv2.putText(
            annotated,
            title,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def parse_pcb_name(value) -> str:
    pcb_name = str(value or "").strip()
    if not pcb_name:
        return "Unnamed PCB"
    return pcb_name[:120]


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def sanitize_supplied_defects(defects_input: List[Dict]) -> List[Dict]:
    sanitized: List[Dict] = []
    for item in defects_input or []:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "unknown")).strip() or "unknown"
        bbox = item.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        try:
            conf = float(item.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        try:
            seen_count = int(item.get("seen_count", 1))
        except Exception:
            seen_count = 1
        seen_count = max(1, seen_count)

        normalized = {
            "label": label,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
            "model": str(item.get("model", "session")).strip() or "session",
            "seen_count": seen_count,
        }
        sanitized.append(normalized)

    # Merge duplicate boxes of same label.
    merged: List[Dict] = []
    for defect in sanitized:
        matched = None
        for candidate in merged:
            if candidate.get("label", "").lower() != defect.get("label", "").lower():
                continue
            if _bbox_iou(candidate["bbox"], defect["bbox"]) >= 0.6:
                matched = candidate
                break
        if matched is None:
            merged.append(defect)
            continue

        matched["confidence"] = max(float(matched.get("confidence", 0.0)), float(defect.get("confidence", 0.0)))
        matched["seen_count"] = int(matched.get("seen_count", 1)) + int(defect.get("seen_count", 1))
        # Keep latest bbox for readability.
        matched["bbox"] = defect["bbox"]

    return merged


def apply_detection_assist(
    defects: List[Dict],
    assist_rows: List[Dict],
) -> Dict[str, Any]:
    review_by_index = {
        int(item.get("index")): item
        for item in (assist_rows or [])
        if isinstance(item, dict) and str(item.get("index", "")).isdigit()
    }

    reviewed_defects: List[Dict] = []
    effective_defects: List[Dict] = []
    filtered_out = 0

    for idx, defect in enumerate(defects or []):
        row = review_by_index.get(idx, {})
        model_conf = float(defect.get("confidence", 0.0))
        ai_conf_raw = row.get("ai_confidence", None)
        ai_conf = None
        if ai_conf_raw is not None:
            try:
                ai_conf = max(0.0, min(1.0, float(ai_conf_raw)))
            except Exception:
                ai_conf = None

        fused_conf = round((model_conf + ai_conf) / 2.0, 4) if ai_conf is not None else model_conf
        ai_verdict = str(row.get("ai_verdict", "UNCERTAIN")).strip().upper() or "UNCERTAIN"
        ai_reason = str(row.get("ai_reason", "")).strip()

        reviewed = dict(defect)
        reviewed["confidence_fused"] = fused_conf
        reviewed["ai_confidence"] = ai_conf
        reviewed["ai_verdict"] = ai_verdict
        reviewed["ai_reason"] = ai_reason
        reviewed_defects.append(reviewed)

        # Conservative filtering: only suppress detections that AI flags as likely false
        # and that also have low model/fused confidence.
        if ai_verdict == "SUSPECT_FALSE_POSITIVE" and model_conf < 0.55 and fused_conf < 0.60:
            filtered_out += 1
            continue
        effective_defects.append(reviewed)

    summary = {
        "enabled": bool(assist_rows),
        "raw_defect_count": len(defects or []),
        "effective_defect_count": len(effective_defects),
        "filtered_out_count": filtered_out,
    }

    return {
        "reviewed_defects": reviewed_defects,
        "effective_defects": effective_defects,
        "summary": summary,
    }


def run_full_pipeline(
    frame,
    source_tag: str = "upload",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_mode: str = None,
    model_key: str = None,
    pcb_name: str = "",
    session_seconds: int = None,
    defects_override: List[Dict] = None,
) -> Dict:
    if detector is None:
        raise RuntimeError(f"Model unavailable: {model_error}")
    ensure_runtime_dirs()

    pcb_name = parse_pcb_name(pcb_name)

    if defects_override is None:
        detection_out = detector.detect(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_mode=model_mode,
            model_key=model_key,
        )
        raw_defects = detection_out.get("defects", []) or []

        detection_assist_rows = gemini_engine.generate_detection_assist(
            {
                "defects": raw_defects,
                "status": detection_out.get("status", "UNKNOWN"),
                "model_mode": detection_out.get("model_mode", ""),
                "models_used": detection_out.get("models_used", []),
            }
        )
        assisted = apply_detection_assist(raw_defects, detection_assist_rows)
        reviewed_defects = assisted["reviewed_defects"]
        defects = assisted["effective_defects"]
        detection_assist = assisted["summary"]
        models_used = detection_out.get("models_used", [])
        models_loaded = detection_out.get("models_loaded", [])
        effective_model_mode = detection_out.get("model_mode", "")
        model_load_errors = detection_out.get("model_load_errors", {})
    else:
        reviewed_defects = sanitize_supplied_defects(defects_override)
        defects = reviewed_defects
        detection_assist = {
            "enabled": True,
            "raw_defect_count": len(reviewed_defects),
            "effective_defect_count": len(defects),
            "filtered_out_count": 0,
            "mode": "session_aggregate",
        }
        models_used = []
        models_loaded = list((getattr(detector, "models", {}) or {}).keys())
        effective_model_mode = "session_aggregate"
        model_load_errors = getattr(detector, "load_errors", {})

    status = "DEFECTIVE" if defects else "OK"

    prediction = predict_failure(defects)
    risk_level = prediction["risk_level"]
    failure_probability = prediction["failure_probability"]

    stamp = now_stamp()
    original_name = f"{source_tag}_{stamp}.jpg"
    detection_name = f"detection_{source_tag}_{stamp}.jpg"
    heatmap_name = f"heatmap_{source_tag}_{stamp}.jpg"
    report_name = f"report_{stamp}.pdf"

    original_path = os.path.join(CAPTURES_DIR, original_name)
    detection_path = os.path.join(DETECTIONS_DIR, detection_name)
    heatmap_path = os.path.join(HEATMAPS_DIR, heatmap_name)
    report_path = os.path.join(REPORTS_DIR, report_name)

    annotated_image = draw_annotated_detections(frame, defects)
    write_image_or_raise(original_path, frame, "original")
    write_image_or_raise(detection_path, annotated_image, "detection")

    heatmap_img = generate_heatmap_overlay(frame, defects)
    save_heatmap(heatmap_path, heatmap_img)

    gemini_payload = {
        "defects": defects,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
        "status": status,
    }
    rule_based_repairs = suggest_repairs(defects)
    repairs = gemini_engine.generate_repair_suggestions(gemini_payload, rule_based_repairs)
    gemini_text = gemini_engine.generate_gemini_explanation(gemini_payload)

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pcb_name": pcb_name,
        "source": source_tag,
        "status": status,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
        # Technical specifications (can be customized per PCB type)
        "operating_voltage": "3.3V-5V (Typical IoT)",
        "max_current": "< 500mA",
        "temp_range": "-20°C to +85°C",
        "communication": "I2C/SPI/UART/WiFi",
        "power_supply": "DC Regulated",
        "compliance": "RoHS/CE/FCC",
        # Component analysis
        "microcontroller": "ESP32/STM32/RP2040",
        "sensors": "Temperature/Humidity/Motion",
        "comm_modules": "WiFi/Bluetooth/Zigbee",
        "power_mgmt": "LDO/Boost Converter",
        "connectors": "USB/JST/Pin Headers",
        "test_points": "TP1-TP8 (Power/GND/Signals)",
        # Quality metrics
        "board_area": "50-100 cm² (estimated)",
    }
    if session_seconds is not None:
        metadata["session_seconds"] = session_seconds

    generate_pdf_report(
        output_path=report_path,
        metadata=metadata,
        defects=defects,
        gemini_text=gemini_text,
        repairs=repairs,
        original_image_path=original_path,
        detection_image_path=detection_path,
        heatmap_image_path=heatmap_path,
    )

    log_record = {
        "timestamp": metadata["timestamp"],
        "source": source_tag,
        "pcb_name": pcb_name,
        "status": status,
        "risk_level": risk_level,
        "failure_probability": str(failure_probability),
        "defects": defects,
        "report_path": report_path,
    }

    log_status = log_store.log_scan(log_record)

    return {
        "timestamp": metadata["timestamp"],
        "status": status,
        "pcb_name": pcb_name,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
        "defects": defects,
        "raw_defects": reviewed_defects,
        "detection_assist": detection_assist,
        "models_used": models_used,
        "models_loaded": models_loaded,
        "model_mode": effective_model_mode,
        "model_load_errors": model_load_errors,
        "repairs": repairs,
        "gemini_explanation": gemini_text,
        "original_image_path": original_path,
        "detection_image_path": detection_path,
        "heatmap_image_path": heatmap_path,
        "original_image_data": encode_ndarray_to_base64(frame),
        "detection_image_data": encode_ndarray_to_base64(annotated_image),
        "heatmap_image_data": encode_ndarray_to_base64(heatmap_img),
        "report_path": report_path,
        "log_status": log_status,
    }


def aggregate_dashboard(logs: List[Dict]) -> Dict:
    total_scans = len(logs)
    defect_counts: Dict[str, int] = {}
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
    status_counts = {"PASS": 0, "FAIL": 0, "UNKNOWN": 0}
    failure_probabilities = []
    defect_density_trend = []
    processing_times = []

    for log in logs:
        risk = str(log.get("risk_level", "UNKNOWN")).upper()
        if risk not in risk_counts:
            risk_counts["UNKNOWN"] += 1
        else:
            risk_counts[risk] += 1

        status = str(log.get("status", "UNKNOWN")).upper()
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["UNKNOWN"] += 1

        for d in log.get("defects", []):
            label = d.get("label", "unknown")
            defect_counts[label] = defect_counts.get(label, 0) + 1

        # Collect failure probabilities for trend analysis
        try:
            prob = float(log.get("failure_probability", 0))
            failure_probabilities.append(prob)
        except (ValueError, TypeError):
            pass

        # Collect defect density (defects per scan)
        defect_count = len(log.get("defects", []))
        defect_density_trend.append(defect_count)

    # Calculate quality metrics
    avg_failure_prob = sum(failure_probabilities) / len(failure_probabilities) if failure_probabilities else 0
    yield_rate = (status_counts["PASS"] / total_scans * 100) if total_scans > 0 else 0
    defect_density_avg = sum(defect_density_trend) / len(defect_density_trend) if defect_density_trend else 0

    recent_reports = [
        {
            "timestamp": log.get("timestamp", ""),
            "pcb_name": log.get("pcb_name", ""),
            "source": log.get("source", "upload"),
            "status": log.get("status", ""),
            "risk_level": log.get("risk_level", ""),
            "failure_probability": log.get("failure_probability", ""),
            "defect_count": len(log.get("defects", [])),
            "report_path": log.get("report_path", ""),
            "report_file": os.path.basename(log.get("report_path", "")) if log.get("report_path") else "",
        }
        for log in logs[:20]
    ]

    return {
        "total_scans": total_scans,
        "defect_distribution": defect_counts,
        "risk_distribution": risk_counts,
        "status_distribution": status_counts,
        "quality_metrics": {
            "average_failure_probability": round(avg_failure_prob, 2),
            "yield_rate": round(yield_rate, 1),
            "average_defect_density": round(defect_density_avg, 2),
            "total_defects": sum(defect_counts.values()),
        },
        "recent_reports": recent_reports,
    }


def parse_float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_model_inspection_details(
    detection_out: Dict,
    requested_model_mode: str,
    requested_model_key: str,
) -> Dict:
    models_loaded = detection_out.get("models_loaded", []) or []
    models_used = detection_out.get("models_used", []) or []
    defects = detection_out.get("defects", []) or []
    by_model: Dict[str, List[Dict]] = {key: [] for key in models_loaded}

    for defect in defects:
        model_key = defect.get("model", "unknown")
        by_model.setdefault(model_key, [])
        by_model[model_key].append(defect)

    per_model = []
    for key in models_loaded:
        model_defects = by_model.get(key, [])
        per_model.append(
            {
                "model_key": key,
                "ran_in_this_scan": key in models_used,
                "defect_count": len(model_defects),
                "defects": model_defects,
            }
        )

    return {
        "requested_mode": requested_model_mode or "default",
        "requested_model_key": requested_model_key or "default",
        "effective_mode": detection_out.get("model_mode", ""),
        "models_loaded": models_loaded,
        "models_used": models_used,
        "model_load_errors": detection_out.get("model_load_errors", {}),
        "per_model": per_model,
    }


def build_all_models_full_results(
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> List[Dict]:
    if detector is None:
        return []

    models_loaded = getattr(detector, "models", {}) or {}
    results: List[Dict] = []
    for model_key in models_loaded.keys():
        try:
            det = detector.detect(
                frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                model_mode="single",
                model_key=model_key,
            )
            defects = det.get("defects", [])
            prediction = predict_failure(defects)
            results.append(
                {
                    "model_key": model_key,
                    "status": det.get("status", "OK"),
                    "defect_count": len(defects),
                    "risk_level": prediction.get("risk_level", "LOW"),
                    "failure_probability": prediction.get("failure_probability", 0),
                    "defects": defects,
                    "annotated_image": encode_ndarray_to_base64(det.get("annotated_image", frame)),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_key": model_key,
                    "error": str(exc),
                    "status": "ERROR",
                    "defect_count": 0,
                    "risk_level": "UNKNOWN",
                    "failure_probability": 0,
                    "defects": [],
                    "annotated_image": "",
                }
            )
    return results


def get_model_catalog() -> List[Dict]:
    if detector is None:
        return []
    output = []
    model_paths = getattr(detector, "model_paths", {}) or {}
    models = getattr(detector, "models", {}) or {}
    for key, model_path in model_paths.items():
        if key not in models:
            continue
        output.append({"key": key, "path": model_path})
    return output


def log_detection_event(source: str, status: str, risk_level: str, failure_probability: float, defects: List[Dict]) -> Dict:
    log_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "status": status,
        "risk_level": risk_level,
        "failure_probability": str(failure_probability),
        "defects": defects,
        "report_path": "",
    }
    return log_store.log_scan(log_record)


def build_report_fields(report_path: str) -> Dict[str, str]:
    report_file = os.path.basename(report_path) if report_path else ""
    report_url = f"/reports/{report_file}" if report_file else ""
    return {
        "report_path": report_path or "",
        "report_file": report_file,
        "report_url": report_url,
    }


def start_live_camera():
    global camera_running

    with camera_lock:
        if camera_running:
            return
        camera_running = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_running = False
        print("[ERROR] Camera not found or inaccessible.")
        return

    print("[INFO] Live camera started. Press 'c' to capture/process, 'q' to quit.")

    while camera_running:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame from camera.")
            break

        display_frame = frame.copy()

        try:
            detection = detector.detect(display_frame) if detector else {"defects": [], "status": "MODEL_ERROR"}
            defects = detection.get("defects", [])
            prediction = predict_failure(defects)

            status = "DEFECTIVE" if defects else "OK"
            risk_level = prediction["risk_level"]
            defect_count = len(defects)

            cv2.putText(display_frame, f"Status: {status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if status == "OK" else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Defects: {defect_count}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Risk: {risk_level}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            for d in defects:
                x1, y1, x2, y2 = d["bbox"]
                label = d["label"]
                conf = d["confidence"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        except Exception as exc:
            cv2.putText(display_frame, f"Detection error: {exc}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("PCB Live Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            try:
                result = run_full_pipeline(frame, source_tag="camera")
                print(
                    f"[CAPTURE] {result['timestamp']} | status={result['status']} | risk={result['risk_level']} "
                    f"| failure_probability={result['failure_probability']}% | report={result['report_path']}"
                )
            except Exception as exc:
                print(f"[ERROR] Capture processing failed: {exc}")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    with camera_lock:
        camera_running = False


@app.route("/")
def index():
    try:
        return render_template(
            "index.html",
            model_error=model_error,
            model_catalog=get_model_catalog(),
        )
    except Exception:
        return jsonify(
            {
                "service": "pcb-detection-backend",
                "status": "ok",
                "message": "Backend is running. Use /health, /detect, /api/live-detect, /api/dashboard",
            }
        )


@app.route("/live")
def live_page():
    return render_template(
        "live.html",
        model_error=model_error,
        model_catalog=get_model_catalog(),
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": detector is not None,
            "model_error": model_error,
            "logging_backend": "local_file",
            "logging_enabled": log_store.enabled,
            "logging_error": log_store.error,
            "gemini_enabled": gemini_engine.enabled,
            "gemini_model": gemini_engine.model_name,
            "model_catalog": get_model_catalog(),
        }
    )


@app.route("/detect", methods=["POST"])
def detect_upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Invalid upload"}), 400

    file_bytes = file.read()
    np_arr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    if np_arr is None:
        return jsonify({"error": "Failed to decode image"}), 400

    conf_threshold = parse_float_or_default(request.form.get("conf_threshold"), 0.25)
    iou_threshold = parse_float_or_default(request.form.get("iou_threshold"), 0.45)
    model_mode = (request.form.get("model_mode") or "").strip().lower() or None
    model_key = (request.form.get("model_key") or "").strip() or None
    pcb_name = parse_pcb_name(request.form.get("pcb_name"))

    try:
        result = run_full_pipeline(
            np_arr,
            source_tag="upload",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_mode=model_mode,
            model_key=model_key,
            pcb_name=pcb_name,
        )
    except (RuntimeError, DetectionError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected processing error: {exc}"}), 500

    response = {
        "timestamp": result["timestamp"],
        "pcb_name": result.get("pcb_name", ""),
        "status": result["status"],
        "risk_level": result["risk_level"],
        "failure_probability": result["failure_probability"],
        "defects": result["defects"],
        "raw_defects": result.get("raw_defects", []),
        "detection_assist": result.get("detection_assist", {}),
        "models_used": result["models_used"],
        "models_loaded": result["models_loaded"],
        "model_mode": result["model_mode"],
        "model_load_errors": result["model_load_errors"],
        "model_inspection_details": build_model_inspection_details(
            {
                "models_loaded": result["models_loaded"],
                "models_used": result["models_used"],
                "model_mode": result["model_mode"],
                "model_load_errors": result["model_load_errors"],
                "defects": result["defects"],
            },
            requested_model_mode=model_mode,
            requested_model_key=model_key,
        ),
        "all_models_full_results": build_all_models_full_results(
            np_arr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        ),
        "config_used": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "model_mode": model_mode or "default",
            "model_key": model_key or "default",
        },
        "repairs": result["repairs"],
        "gemini_explanation": result["gemini_explanation"],
        "report_path": result["report_path"],
        "original_image": encode_image_to_base64_safe(
            result["original_image_path"],
            result.get("original_image_data", ""),
        ),
        "detection_image": encode_image_to_base64_safe(
            result["detection_image_path"],
            result.get("detection_image_data", ""),
        ),
        "heatmap_image": encode_image_to_base64_safe(
            result["heatmap_image_path"],
            result.get("heatmap_image_data", ""),
        ),
        "log_status": result["log_status"],
    }
    return jsonify(response)


@app.route("/api/live-detect", methods=["POST"])
def live_detect():
    if detector is None:
        return jsonify({"error": f"Model unavailable: {model_error}"}), 500

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image_base64", "")
    if not image_data:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        frame = decode_data_url_to_image(image_data)
        conf_threshold = parse_float_or_default(payload.get("conf_threshold"), 0.25)
        iou_threshold = parse_float_or_default(payload.get("iou_threshold"), 0.45)
        model_mode = (payload.get("model_mode") or "").strip().lower() or None
        model_key = (payload.get("model_key") or "").strip() or None
        pcb_name = parse_pcb_name(payload.get("pcb_name"))

        should_log = str(payload.get("log_event", "")).strip().lower() in {"1", "true", "yes", "on"}

        if should_log:
            full_result = run_full_pipeline(
                frame,
                source_tag="live_stream",
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                model_mode=model_mode,
                model_key=model_key,
                pcb_name=pcb_name,
            )
            report_fields = build_report_fields(full_result.get("report_path", ""))

            return jsonify(
                {
                    "status": full_result.get("status", "OK"),
                    "pcb_name": full_result.get("pcb_name", ""),
                    "defect_count": len(full_result.get("defects", [])),
                    "risk_level": full_result.get("risk_level", "LOW"),
                    "failure_probability": full_result.get("failure_probability", 0),
                    "defects": full_result.get("defects", []),
                    "raw_defects": full_result.get("raw_defects", []),
                    "detection_assist": full_result.get("detection_assist", {}),
                    "models_used": full_result.get("models_used", []),
                    "models_loaded": full_result.get("models_loaded", []),
                    "model_mode": full_result.get("model_mode", ""),
                    "live_logged": True,
                    "report_generated": bool(report_fields.get("report_file")),
                    "log_status": full_result.get("log_status", {"success": False, "error": "log unavailable"}),
                    "config_used": {
                        "conf_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "model_mode": model_mode or "default",
                        "model_key": model_key or "default",
                    },
                    **report_fields,
                }
            )

        detection_out = detector.detect(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_size=960,
            model_mode=model_mode,
            model_key=model_key,
        )
        defects = detection_out.get("defects", [])
        prediction = predict_failure(defects)
        risk_level = prediction.get("risk_level", "LOW")
        failure_probability = prediction.get("failure_probability", 0)

        return jsonify(
            {
                "status": detection_out.get("status", "OK"),
                "pcb_name": pcb_name,
                "defect_count": len(defects),
                "risk_level": risk_level,
                "failure_probability": failure_probability,
                "defects": defects,
                "models_used": detection_out.get("models_used", []),
                "models_loaded": detection_out.get("models_loaded", []),
                "model_mode": detection_out.get("model_mode", ""),
                "live_logged": False,
                "report_generated": False,
                "log_status": {"success": False, "error": "logging disabled"},
                "detection_assist": {
                    "enabled": False,
                    "raw_defect_count": len(defects),
                    "effective_defect_count": len(defects),
                    "filtered_out_count": 0,
                },
                "config_used": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "model_mode": model_mode or "default",
                    "model_key": model_key or "default",
                },
                **build_report_fields(""),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Live detection failed: {exc}"}), 500


@app.route("/api/live-detect-upload", methods=["POST"])
def live_detect_upload():
    if detector is None:
        return jsonify({"error": f"Model unavailable: {model_error}"}), 500

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "image file is required"}), 400

    try:
        file_bytes = file.read()
        frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode uploaded frame"}), 400

        conf_threshold = parse_float_or_default(request.form.get("conf_threshold"), 0.25)
        iou_threshold = parse_float_or_default(request.form.get("iou_threshold"), 0.45)
        model_mode = (request.form.get("model_mode") or "").strip().lower() or None
        model_key = (request.form.get("model_key") or "").strip() or None
        pcb_name = parse_pcb_name(request.form.get("pcb_name"))

        should_log = str(request.form.get("log_event", "")).strip().lower() in {"1", "true", "yes", "on"}

        if should_log:
            full_result = run_full_pipeline(
                frame,
                source_tag="live_stream",
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                model_mode=model_mode,
                model_key=model_key,
                pcb_name=pcb_name,
            )
            report_fields = build_report_fields(full_result.get("report_path", ""))

            return jsonify(
                {
                    "status": full_result.get("status", "OK"),
                    "pcb_name": full_result.get("pcb_name", ""),
                    "defect_count": len(full_result.get("defects", [])),
                    "risk_level": full_result.get("risk_level", "LOW"),
                    "failure_probability": full_result.get("failure_probability", 0),
                    "defects": full_result.get("defects", []),
                    "raw_defects": full_result.get("raw_defects", []),
                    "detection_assist": full_result.get("detection_assist", {}),
                    "models_used": full_result.get("models_used", []),
                    "models_loaded": full_result.get("models_loaded", []),
                    "model_mode": full_result.get("model_mode", ""),
                    "live_logged": True,
                    "report_generated": bool(report_fields.get("report_file")),
                    "log_status": full_result.get("log_status", {"success": False, "error": "log unavailable"}),
                    "config_used": {
                        "conf_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "model_mode": model_mode or "default",
                        "model_key": model_key or "default",
                    },
                    **report_fields,
                }
            )

        detection_out = detector.detect(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_size=960,
            model_mode=model_mode,
            model_key=model_key,
        )
        defects = detection_out.get("defects", [])
        prediction = predict_failure(defects)
        risk_level = prediction.get("risk_level", "LOW")
        failure_probability = prediction.get("failure_probability", 0)

        return jsonify(
            {
                "status": detection_out.get("status", "OK"),
                "pcb_name": pcb_name,
                "defect_count": len(defects),
                "risk_level": risk_level,
                "failure_probability": failure_probability,
                "defects": defects,
                "models_used": detection_out.get("models_used", []),
                "models_loaded": detection_out.get("models_loaded", []),
                "model_mode": detection_out.get("model_mode", ""),
                "live_logged": False,
                "report_generated": False,
                "log_status": {"success": False, "error": "logging disabled"},
                "detection_assist": {
                    "enabled": False,
                    "raw_defect_count": len(defects),
                    "effective_defect_count": len(defects),
                    "filtered_out_count": 0,
                },
                "config_used": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "model_mode": model_mode or "default",
                    "model_key": model_key or "default",
                },
                **build_report_fields(""),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Live detection upload failed: {exc}"}), 500


@app.route("/api/live-session-report", methods=["POST"])
def live_session_report():
    if detector is None:
        return jsonify({"error": f"Model unavailable: {model_error}"}), 500

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "image file is required"}), 400

    try:
        file_bytes = file.read()
        frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode uploaded frame"}), 400

        conf_threshold = parse_float_or_default(request.form.get("conf_threshold"), 0.25)
        iou_threshold = parse_float_or_default(request.form.get("iou_threshold"), 0.45)
        model_mode = (request.form.get("model_mode") or "").strip().lower() or None
        model_key = (request.form.get("model_key") or "").strip() or None
        pcb_name = parse_pcb_name(request.form.get("pcb_name"))

        try:
            session_seconds = int(float(request.form.get("session_seconds", 60)))
        except Exception:
            session_seconds = 60
        session_seconds = max(1, min(600, session_seconds))

        has_aggregated_payload = "aggregated_defects" in request.form
        raw_agg = request.form.get("aggregated_defects", "[]")
        parsed_aggregated_ok = True
        try:
            aggregated_input = json.loads(raw_agg) if raw_agg else []
        except json.JSONDecodeError:
            aggregated_input = []
            parsed_aggregated_ok = False

        aggregated_defects = sanitize_supplied_defects(aggregated_input)
        # Keep explicit empty session payload as-is (valid "no defects in this 1-minute session").
        # Fallback to a single-frame detect only if payload is missing or invalid.
        if not aggregated_defects and (not has_aggregated_payload or not parsed_aggregated_ok):
            detection_out = detector.detect(
                frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_size=960,
                model_mode=model_mode,
                model_key=model_key,
            )
            aggregated_defects = sanitize_supplied_defects(detection_out.get("defects", []))

        result = run_full_pipeline(
            frame,
            source_tag="live_session",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_mode=model_mode,
            model_key=model_key,
            pcb_name=pcb_name,
            session_seconds=session_seconds,
            defects_override=aggregated_defects,
        )
        report_fields = build_report_fields(result.get("report_path", ""))

        return jsonify(
            {
                "timestamp": result.get("timestamp", ""),
                "pcb_name": result.get("pcb_name", ""),
                "session_seconds": session_seconds,
                "status": result.get("status", "OK"),
                "defect_count": len(result.get("defects", [])),
                "risk_level": result.get("risk_level", "LOW"),
                "failure_probability": result.get("failure_probability", 0),
                "defects": result.get("defects", []),
                "raw_defects": result.get("raw_defects", []),
                "detection_assist": result.get("detection_assist", {}),
                "models_used": result.get("models_used", []),
                "models_loaded": result.get("models_loaded", []),
                "model_mode": result.get("model_mode", ""),
                "log_status": result.get("log_status", {"success": False, "error": "log unavailable"}),
                "report_generated": bool(report_fields.get("report_file")),
                "gemini_explanation": result.get("gemini_explanation", ""),
                "repairs": result.get("repairs", []),
                "config_used": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "model_mode": model_mode or "default",
                    "model_key": model_key or "default",
                },
                **report_fields,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Live session report failed: {exc}"}), 500


@app.route("/start-live-camera", methods=["POST"])
def start_camera_route():
    global camera_thread

    if detector is None:
        return jsonify({"error": f"Model unavailable: {model_error}"}), 500

    with camera_lock:
        if camera_running:
            return jsonify({"message": "Live camera is already running."})

        camera_thread = threading.Thread(target=start_live_camera, daemon=True)
        camera_thread.start()

    return jsonify(
        {
            "message": "Legacy OpenCV server-camera mode started. Prefer browser live mode on main page.",
        }
    )


@app.route("/dashboard")
def dashboard():
    logs = log_store.fetch_logs(limit=300)

    analytics = aggregate_dashboard(logs)
    return render_template("dashboard.html", analytics=analytics)


@app.route("/api/dashboard")
def dashboard_api():
    logs = log_store.fetch_logs(limit=300)
    analytics = aggregate_dashboard(logs)
    return jsonify(analytics)


@app.route("/reports/<path:filename>")
def report_file(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
