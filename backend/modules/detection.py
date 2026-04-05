import os
from threading import Lock
from typing import Dict, List, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class DetectionError(Exception):
    """Raised when detection cannot be completed."""


class YOLODetector:
    """Singleton-style YOLO detector that loads available models only once."""

    _instance = None
    _lock = Lock()

    def __new__(cls, model_path: str = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str = None):
        if self._initialized:
            return

        self.models: Dict[str, YOLO] = {}
        self.model_paths = self._resolve_model_paths(model_path)
        self.load_errors: Dict[str, str] = {}

        for key, path in self.model_paths.items():
            try:
                self.models[key] = YOLO(path)
            except Exception as exc:
                self.load_errors[key] = str(exc)

        if not self.models:
            raise FileNotFoundError(
                "No YOLO model could be loaded. Expected one or more model files in backend/yolo_model."
            )

        self.default_model_key = os.getenv("YOLO_MODEL_KEY", "model").strip() or "model"
        if self.default_model_key not in self.models:
            self.default_model_key = next(iter(self.models.keys()))

        self.default_mode = os.getenv("YOLO_INFERENCE_MODE", "all").strip().lower()
        if self.default_mode not in {"single", "all"}:
            self.default_mode = "all"

        self._initialized = True

    @staticmethod
    def _resolve_model_paths(model_path: str = None) -> Dict[str, str]:
        module_file = Path(__file__).resolve()
        app_dir = module_file.parent.parent  # .../backend
        workspace_dir = app_dir.parent        # .../PCB DETECTION

        candidates = {
            "model": str(app_dir / "yolo_model" / "model.pt"),
            "full": str(app_dir / "yolo_model" / "yolo_full_pcb.pt"),
            "bare": str(app_dir / "yolo_model" / "yolo_bare_pcb2.pt"),
        }

        if model_path and os.path.exists(model_path):
            candidates["custom"] = model_path

        env_path = os.getenv("YOLO_MODEL_PATH", "").strip()
        if env_path and os.path.exists(env_path):
            candidates["env"] = env_path

        root_candidates = {
            "full_root": str(workspace_dir / "yolo_full_pcb.pt"),
            "bare_root": str(workspace_dir / "yolo_bare_pcb2.pt"),
        }
        for key, path in root_candidates.items():
            if os.path.exists(path):
                candidates[key] = path

        existing = {key: path for key, path in candidates.items() if os.path.exists(path)}
        return existing

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_size: int = 1280,
        model_mode: str = None,
        model_key: str = None,
    ) -> Dict[str, Any]:
        if image is None or image.size == 0:
            raise DetectionError("Invalid image provided for detection")

        original_h, original_w = image.shape[:2]
        resized, scale = self._resize_for_speed(image, max_size=max_size)

        selected_mode = (model_mode or self.default_mode or "all").lower()
        if selected_mode not in {"single", "all"}:
            selected_mode = "all"

        selected_model_key = (model_key or self.default_model_key).strip() if (model_key or self.default_model_key) else self.default_model_key
        if selected_model_key not in self.models:
            selected_model_key = self.default_model_key

        run_keys = [selected_model_key] if selected_mode == "single" else list(self.models.keys())

        defects: List[Dict[str, Any]] = []
        for key in run_keys:
            model = self.models.get(key)
            if model is None:
                continue
            try:
                defects.extend(
                    self._predict_with_model(
                        model=model,
                        model_name=key,
                        image=resized,
                        scale=scale,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                    )
                )
            except Exception:
                # Keep processing remaining models.
                continue

        defects = self._dedupe_defects(defects)
        annotated = self._draw_detections(image.copy(), defects)

        status = "DEFECTIVE" if defects else "OK"

        return {
            "defects": defects,
            "status": status,
            "defect_count": len(defects),
            "annotated_image": annotated,
            "image_shape": (original_h, original_w),
            "model_mode": selected_mode,
            "models_used": run_keys,
            "models_loaded": list(self.models.keys()),
            "model_load_errors": self.load_errors,
        }

    @staticmethod
    def _predict_with_model(
        model: YOLO,
        model_name: str,
        image: np.ndarray,
        scale: float,
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        output: List[Dict[str, Any]] = []
        if result.boxes is None or len(result.boxes) == 0:
            return output

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            x1, y1, x2, y2 = [
                int(x1 / scale),
                int(y1 / scale),
                int(x2 / scale),
                int(y2 / scale),
            ]

            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            output.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2],
                    "model": model_name,
                }
            )
        return output

    @staticmethod
    def _draw_detections(image: np.ndarray, defects: List[Dict[str, Any]]) -> np.ndarray:
        for defect in defects:
            x1, y1, x2, y2 = defect["bbox"]
            label = defect.get("label", "unknown")
            confidence = defect.get("confidence", 0.0)
            source_model = defect.get("model", "")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 140, 255), 2)
            title = f"{label} {confidence:.2f}"
            if source_model:
                title = f"{title} [{source_model}]"
            cv2.putText(
                image,
                title,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
        return image

    @staticmethod
    def _dedupe_defects(defects: List[Dict[str, Any]], iou_threshold: float = 0.65) -> List[Dict[str, Any]]:
        if not defects:
            return []

        sorted_defects = sorted(defects, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        kept: List[Dict[str, Any]] = []

        for defect in sorted_defects:
            overlap = False
            for chosen in kept:
                if defect.get("label", "").lower() != chosen.get("label", "").lower():
                    continue
                if YOLODetector._bbox_iou(defect["bbox"], chosen["bbox"]) >= iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(defect)

        return kept

    @staticmethod
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

    @staticmethod
    def _resize_for_speed(image: np.ndarray, max_size: int = 1280) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= max_size:
            return image, 1.0

        scale = max_size / float(max_dim)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
