import base64
import os
import threading
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from dotenv import load_dotenv

from modules.detection import YOLODetector, DetectionError
from modules.failure_prediction import predict_failure
from modules.firebase_db import FirebaseLogger
from modules.gemini_ai import GeminiEngine
from modules.heatmap import generate_heatmap_overlay, save_heatmap
from modules.repair_engine import suggest_repairs
from modules.report_generator import generate_pdf_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)


def load_environment():
    env_candidates = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(WORKSPACE_DIR, "pcb_system", ".env"),
    ]
    for env_path in env_candidates:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)

    # Optional fallback for local setups that only have values in .env.example.
    if not (os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip() and os.getenv("FIREBASE_DB_URL", "").strip()):
        example_path = os.path.join(BASE_DIR, ".env.example")
        if os.path.exists(example_path):
            load_dotenv(example_path, override=False)


load_environment()

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
DETECTIONS_DIR = os.path.join(BASE_DIR, "detections")
HEATMAPS_DIR = os.path.join(BASE_DIR, "heatmaps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for d in [CAPTURES_DIR, DETECTIONS_DIR, HEATMAPS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")


detector = None
model_error = None
try:
    detector = YOLODetector()
except Exception as exc:
    model_error = str(exc)

gemini_engine = GeminiEngine()
firebase_logger = FirebaseLogger()

camera_thread = None
camera_running = False
camera_lock = threading.Lock()


LOCAL_FALLBACK_LOGS: List[Dict] = []


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


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


def run_full_pipeline(
    frame,
    source_tag: str = "upload",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_mode: str = None,
    model_key: str = None,
) -> Dict:
    if detector is None:
        raise RuntimeError(f"Model unavailable: {model_error}")

    detection_out = detector.detect(
        frame,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        model_mode=model_mode,
        model_key=model_key,
    )
    defects = detection_out["defects"]
    status = detection_out["status"]

    prediction = predict_failure(defects)
    risk_level = prediction["risk_level"]
    failure_probability = prediction["failure_probability"]

    repairs = suggest_repairs(defects)

    stamp = now_stamp()
    original_name = f"{source_tag}_{stamp}.jpg"
    detection_name = f"detection_{source_tag}_{stamp}.jpg"
    heatmap_name = f"heatmap_{source_tag}_{stamp}.jpg"
    report_name = f"report_{stamp}.pdf"

    original_path = os.path.join(CAPTURES_DIR, original_name)
    detection_path = os.path.join(DETECTIONS_DIR, detection_name)
    heatmap_path = os.path.join(HEATMAPS_DIR, heatmap_name)
    report_path = os.path.join(REPORTS_DIR, report_name)

    cv2.imwrite(original_path, frame)
    cv2.imwrite(detection_path, detection_out.get("annotated_image", frame))

    heatmap_img = generate_heatmap_overlay(frame, defects)
    save_heatmap(heatmap_path, heatmap_img)

    gemini_payload = {
        "defects": defects,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
        "status": status,
    }
    gemini_text = gemini_engine.generate_gemini_explanation(gemini_payload)

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
    }

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
        "status": status,
        "risk_level": risk_level,
        "failure_probability": str(failure_probability),
        "defects": defects,
        "report_path": report_path,
    }

    fb_status = firebase_logger.log_scan(log_record)
    if not fb_status.get("success"):
        LOCAL_FALLBACK_LOGS.append(log_record)

    return {
        "timestamp": metadata["timestamp"],
        "status": status,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
        "defects": defects,
        "models_used": detection_out.get("models_used", []),
        "models_loaded": detection_out.get("models_loaded", []),
        "model_mode": detection_out.get("model_mode", ""),
        "model_load_errors": detection_out.get("model_load_errors", {}),
        "repairs": repairs,
        "gemini_explanation": gemini_text,
        "original_image_path": original_path,
        "detection_image_path": detection_path,
        "heatmap_image_path": heatmap_path,
        "report_path": report_path,
        "firebase_status": fb_status,
    }


def aggregate_dashboard(logs: List[Dict]) -> Dict:
    total_scans = len(logs)
    defect_counts: Dict[str, int] = {}
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}

    for log in logs:
        risk = str(log.get("risk_level", "UNKNOWN")).upper()
        if risk not in risk_counts:
            risk_counts["UNKNOWN"] += 1
        else:
            risk_counts[risk] += 1

        for d in log.get("defects", []):
            label = d.get("label", "unknown")
            defect_counts[label] = defect_counts.get(label, 0) + 1

    recent_reports = [
        {
            "timestamp": log.get("timestamp", ""),
            "source": log.get("source", "upload"),
            "status": log.get("status", ""),
            "risk_level": log.get("risk_level", ""),
            "report_path": log.get("report_path", ""),
            "report_file": os.path.basename(log.get("report_path", "")) if log.get("report_path") else "",
        }
        for log in logs[:20]
    ]

    return {
        "total_scans": total_scans,
        "defect_distribution": defect_counts,
        "risk_distribution": risk_counts,
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


def merge_logs(remote_logs: List[Dict], local_logs: List[Dict], limit: int = 300) -> List[Dict]:
    merged = list(remote_logs or []) + list(local_logs or [])
    merged.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return merged[:limit]


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
    fb_status = firebase_logger.log_scan(log_record)
    if not fb_status.get("success"):
        LOCAL_FALLBACK_LOGS.append(log_record)
    return fb_status


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
            firebase_error=firebase_logger.error,
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
        firebase_error=firebase_logger.error,
        model_catalog=get_model_catalog(),
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": detector is not None,
            "model_error": model_error,
            "firebase_enabled": firebase_logger.enabled,
            "firebase_error": firebase_logger.error,
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

    try:
        result = run_full_pipeline(
            np_arr,
            source_tag="upload",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_mode=model_mode,
            model_key=model_key,
        )
    except (RuntimeError, DetectionError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected processing error: {exc}"}), 500

    response = {
        "timestamp": result["timestamp"],
        "status": result["status"],
        "risk_level": result["risk_level"],
        "failure_probability": result["failure_probability"],
        "defects": result["defects"],
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
        "original_image": encode_image_to_base64(result["original_image_path"]),
        "detection_image": encode_image_to_base64(result["detection_image_path"]),
        "heatmap_image": encode_image_to_base64(result["heatmap_image_path"]),
        "firebase_status": result["firebase_status"],
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
        should_log = str(payload.get("log_event", "")).strip().lower() in {"1", "true", "yes", "on"}
        fb_status = log_detection_event(
            source="live_stream",
            status=detection_out.get("status", "OK"),
            risk_level=risk_level,
            failure_probability=failure_probability,
            defects=defects,
        ) if should_log else {"success": False, "error": "logging disabled"}

        return jsonify(
            {
                "status": detection_out.get("status", "OK"),
                "defect_count": len(defects),
                "risk_level": risk_level,
                "failure_probability": failure_probability,
                "defects": defects,
                "models_used": detection_out.get("models_used", []),
                "models_loaded": detection_out.get("models_loaded", []),
                "model_mode": detection_out.get("model_mode", ""),
                "live_logged": should_log,
                "firebase_status": fb_status,
                "config_used": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "model_mode": model_mode or "default",
                    "model_key": model_key or "default",
                },
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
        should_log = str(request.form.get("log_event", "")).strip().lower() in {"1", "true", "yes", "on"}
        fb_status = log_detection_event(
            source="live_stream",
            status=detection_out.get("status", "OK"),
            risk_level=risk_level,
            failure_probability=failure_probability,
            defects=defects,
        ) if should_log else {"success": False, "error": "logging disabled"}

        return jsonify(
            {
                "status": detection_out.get("status", "OK"),
                "defect_count": len(defects),
                "risk_level": risk_level,
                "failure_probability": failure_probability,
                "defects": defects,
                "models_used": detection_out.get("models_used", []),
                "models_loaded": detection_out.get("models_loaded", []),
                "model_mode": detection_out.get("model_mode", ""),
                "live_logged": should_log,
                "firebase_status": fb_status,
                "config_used": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "model_mode": model_mode or "default",
                    "model_key": model_key or "default",
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Live detection upload failed: {exc}"}), 500


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
    logs = merge_logs(
        firebase_logger.fetch_logs(limit=300),
        sorted(LOCAL_FALLBACK_LOGS, key=lambda x: x.get("timestamp", ""), reverse=True),
        limit=300,
    )

    analytics = aggregate_dashboard(logs)
    return render_template("dashboard.html", analytics=analytics)


@app.route("/api/dashboard")
def dashboard_api():
    logs = merge_logs(
        firebase_logger.fetch_logs(limit=300),
        sorted(LOCAL_FALLBACK_LOGS, key=lambda x: x.get("timestamp", ""), reverse=True),
        limit=300,
    )
    analytics = aggregate_dashboard(logs)
    return jsonify(analytics)


@app.route("/reports/<path:filename>")
def report_file(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
