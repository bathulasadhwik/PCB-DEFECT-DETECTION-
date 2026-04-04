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
load_dotenv(os.path.join(BASE_DIR, ".env"))

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
HEATMAPS_DIR = os.path.join(BASE_DIR, "heatmaps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for d in [CAPTURES_DIR, HEATMAPS_DIR, REPORTS_DIR]:
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


def run_full_pipeline(frame, source_tag: str = "upload") -> Dict:
    if detector is None:
        raise RuntimeError(f"Model unavailable: {model_error}")

    detection_out = detector.detect(frame)
    defects = detection_out["defects"]
    status = detection_out["status"]

    prediction = predict_failure(defects)
    risk_level = prediction["risk_level"]
    failure_probability = prediction["failure_probability"]

    repairs = suggest_repairs(defects)

    stamp = now_stamp()
    original_name = f"{source_tag}_{stamp}.jpg"
    heatmap_name = f"heatmap_{source_tag}_{stamp}.jpg"
    report_name = f"report_{stamp}.pdf"

    original_path = os.path.join(CAPTURES_DIR, original_name)
    heatmap_path = os.path.join(HEATMAPS_DIR, heatmap_name)
    report_path = os.path.join(REPORTS_DIR, report_name)

    cv2.imwrite(original_path, frame)

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
        heatmap_image_path=heatmap_path,
    )

    log_record = {
        "timestamp": metadata["timestamp"],
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
        return render_template("index.html", model_error=model_error, firebase_error=firebase_logger.error)
    except Exception:
        return jsonify(
            {
                "service": "pcb-detection-backend",
                "status": "ok",
                "message": "Backend is running. Use /health, /detect, /api/live-detect, /api/dashboard",
            }
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

    try:
        result = run_full_pipeline(np_arr, source_tag="upload")
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
        "repairs": result["repairs"],
        "gemini_explanation": result["gemini_explanation"],
        "report_path": result["report_path"],
        "original_image": encode_image_to_base64(result["original_image_path"]),
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
        detection_out = detector.detect(frame, max_size=960)
        defects = detection_out.get("defects", [])
        prediction = predict_failure(defects)

        return jsonify(
            {
                "status": detection_out.get("status", "OK"),
                "defect_count": len(defects),
                "risk_level": prediction.get("risk_level", "LOW"),
                "failure_probability": prediction.get("failure_probability", 0),
                "defects": defects,
                "models_used": detection_out.get("models_used", []),
                "models_loaded": detection_out.get("models_loaded", []),
                "model_mode": detection_out.get("model_mode", ""),
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

        detection_out = detector.detect(frame, max_size=960)
        defects = detection_out.get("defects", [])
        prediction = predict_failure(defects)

        return jsonify(
            {
                "status": detection_out.get("status", "OK"),
                "defect_count": len(defects),
                "risk_level": prediction.get("risk_level", "LOW"),
                "failure_probability": prediction.get("failure_probability", 0),
                "defects": defects,
                "models_used": detection_out.get("models_used", []),
                "models_loaded": detection_out.get("models_loaded", []),
                "model_mode": detection_out.get("model_mode", ""),
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
    logs = firebase_logger.fetch_logs(limit=300)
    if not logs:
        logs = sorted(LOCAL_FALLBACK_LOGS, key=lambda x: x.get("timestamp", ""), reverse=True)

    analytics = aggregate_dashboard(logs)
    return render_template("dashboard.html", analytics=analytics)


@app.route("/api/dashboard")
def dashboard_api():
    logs = firebase_logger.fetch_logs(limit=300)
    if not logs:
        logs = sorted(LOCAL_FALLBACK_LOGS, key=lambda x: x.get("timestamp", ""), reverse=True)
    analytics = aggregate_dashboard(logs)
    return jsonify(analytics)


@app.route("/reports/<path:filename>")
def report_file(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
