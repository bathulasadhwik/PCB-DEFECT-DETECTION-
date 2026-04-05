import os
import json
import re
from typing import Dict, List


DEFAULT_EXPLANATION = (
    "AI explanation unavailable. Review defect confidence, perform electrical validation, "
    "and apply recommended repairs before production release."
)


class GeminiEngine:
    def __init__(self):
        raw_key = os.getenv("GEMINI_API_KEY", "").strip()
        placeholder_keys = {
            "",
            "your_gemini_api_key_here",
            "change_me",
            "changeme",
            "replace_me",
        }
        self.api_key = "" if raw_key.lower() in placeholder_keys else raw_key
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.fallback_models = self._build_fallback_models(self.model_name)
        self.enabled = bool(self.api_key)
        self._client = None
        self.last_error = ""

        if self.enabled:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except Exception as exc:
                self.enabled = False
                self.last_error = f"Gemini SDK init failed: {exc}"
        else:
            self.last_error = "GEMINI_API_KEY is missing or placeholder."

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        model = (name or "").strip()
        if model.startswith("models/"):
            model = model.split("/", 1)[1].strip()
        return model

    def _build_fallback_models(self, preferred: str) -> List[str]:
        ordered: List[str] = []

        def add(name: str):
            n = self._normalize_model_name(name)
            if n and n not in ordered:
                ordered.append(n)

        add(preferred)
        # Keep multiple candidates to survive model deprecations.
        add("gemini-1.5-flash")
        add("gemini-2.0-flash")
        add("gemini-2.5-flash")
        return ordered

    @staticmethod
    def _to_plain_text(text: str) -> str:
        cleaned_lines = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Remove markdown symbols and preserve readable numbered output.
            line = line.replace("**", "").replace("*", "").replace("`", "")
            if line.startswith("#"):
                line = line.lstrip("#").strip()
            if line.startswith("- "):
                line = line[2:].strip()
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _extract_json_array(text: str):
        if not text:
            return None
        cleaned = text.strip()
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return None
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None

    def _normalize_repairs(self, repairs: List[Dict]) -> List[Dict]:
        normalized: List[Dict] = []
        seen = set()

        for item in repairs or []:
            if not isinstance(item, dict):
                continue

            defect = str(
                item.get("defect")
                or item.get("issue")
                or item.get("label")
                or "unknown"
            ).strip()
            suggestion = str(
                item.get("suggestion")
                or item.get("action")
                or item.get("repair")
                or ""
            ).strip()
            if not suggestion:
                continue

            record = {
                "defect": self._to_plain_text(defect) or "unknown",
                "suggestion": self._to_plain_text(suggestion),
            }

            priority = str(item.get("priority") or "").strip().upper()
            if priority in {"HIGH", "MEDIUM", "LOW"}:
                record["priority"] = priority

            validation = str(item.get("validation") or item.get("verification") or "").strip()
            if validation:
                record["validation"] = self._to_plain_text(validation)

            key = (record["defect"].lower(), record["suggestion"].lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(record)

        return normalized

    def _normalize_detection_audit(self, audits: List[Dict], total_defects: int) -> List[Dict]:
        normalized: List[Dict] = []
        seen_indexes = set()

        for item in audits or []:
            if not isinstance(item, dict):
                continue

            try:
                idx = int(item.get("index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= total_defects or idx in seen_indexes:
                continue

            raw_verdict = str(
                item.get("ai_verdict")
                or item.get("verdict")
                or item.get("decision")
                or "UNCERTAIN"
            ).strip().upper()

            if raw_verdict in {"CONFIRMED_TRUE", "TRUE_DEFECT", "CONFIRMED"}:
                ai_verdict = "CONFIRMED"
            elif raw_verdict in {"FALSE", "FALSE_POSITIVE", "SUSPECT_FALSE", "SUSPECT_FALSE_POSITIVE"}:
                ai_verdict = "SUSPECT_FALSE_POSITIVE"
            else:
                ai_verdict = "UNCERTAIN"

            ai_confidence = None
            try:
                ai_confidence = float(item.get("ai_confidence"))
                ai_confidence = max(0.0, min(1.0, ai_confidence))
            except Exception:
                ai_confidence = None

            reason = self._to_plain_text(str(item.get("reason", "")).strip())

            normalized.append(
                {
                    "index": idx,
                    "ai_verdict": ai_verdict,
                    "ai_confidence": ai_confidence,
                    "ai_reason": reason,
                }
            )
            seen_indexes.add(idx)

        return normalized

    def _build_structured_fallback_explanation(self, data: Dict) -> str:
        defects: List[Dict] = data.get("defects", [])
        risk_level = str(data.get("risk_level", "UNKNOWN")).upper()
        failure_probability = data.get("failure_probability", "N/A")
        status = str(data.get("status", "UNKNOWN")).upper()
        reason_line = self._to_plain_text(self.last_error or "Gemini response unavailable.")

        if not defects:
            return (
                "1) Inspection Status\n"
                f"- Overall status: {status}\n"
                f"- Risk level: {risk_level}\n"
                f"- Predicted failure probability: {failure_probability}%\n\n"
                "2) Defect Assessment\n"
                "- No defects were reported by the current detection pipeline.\n\n"
                "3) Operational Risk Impact\n"
                "- No immediate production blocker detected from visual defects.\n"
                "- Continue with electrical validation and burn-in sampling as per QA plan.\n\n"
                "4) Recommended Actions\n"
                "- Perform continuity and isolation checks before release.\n"
                "- Run AOI re-check on a sample batch to validate process stability.\n\n"
                "5) Engineering Notes\n"
                "- Absence of detected defects does not guarantee zero defects; undetected issues may exist below detection thresholds.\n"
                "- Electrical validation is critical to confirm functionality, especially for high-reliability applications.\n"
                "- Burn-in sampling helps identify early-life failures that visual inspection cannot detect.\n"
                "- Process stability validation ensures consistent quality across production batches.\n"
                f"- Gemini unavailable reason: {reason_line}\n"
                "- This fallback explanation is generated without Gemini response."
            )

        defect_lines = []
        for idx, defect in enumerate(defects, start=1):
            label = str(defect.get("label", "unknown"))
            conf = float(defect.get("confidence", 0.0))
            fused = defect.get("confidence_fused", conf)
            try:
                fused = float(fused)
            except Exception:
                fused = conf
            bbox = defect.get("bbox", [])
            defect_lines.append(
                f"- Defect {idx}: {label} | model_conf={conf:.2f} | fused_conf={fused:.2f} | bbox={bbox}"
            )

        if risk_level == "HIGH":
            ops_risk = "- High probability of field failure if unreworked defects remain."
            release_note = "- Hold production release until full rework and electrical validation are complete."
        elif risk_level == "MEDIUM":
            ops_risk = "- Moderate reliability risk; latent failures are possible under thermal/cycle stress."
            release_note = "- Release only after targeted rework plus validation on critical nets."
        else:
            ops_risk = "- Low immediate risk, but confirm no critical net impact from detected defects."
            release_note = "- Controlled release is possible after verification checks pass."

        return (
            "1) Inspection Status\n"
            f"- Overall status: {status}\n"
            f"- Risk level: {risk_level}\n"
            f"- Predicted failure probability: {failure_probability}%\n\n"
            "2) Defect Assessment\n"
            + "\n".join(defect_lines)
            + "\n\n3) Operational Risk Impact\n"
            f"{ops_risk}\n\n"
            "4) Recommended Actions\n"
            "- Rework defects by label priority (short/open/missing first).\n"
            "- Reinspect repaired regions under magnification and AOI.\n"
            "- Perform continuity + isolation checks on affected paths.\n\n"
            "5) Release Decision Guidance\n"
            f"{release_note}\n"
            "- Full rework includes repairing all detected defects with appropriate techniques (soldering, component replacement, trace repair).\n"
            "- Electrical validation encompasses continuity testing, isolation checks, and functional testing under operational conditions.\n"
            "- Release without validation risks field failures, warranty claims, and potential safety issues in critical applications.\n"
            "- Validation timeline should include thermal cycling and stress testing to ensure long-term reliability.\n"
            f"- Gemini unavailable reason: {reason_line}\n"
            "- This fallback explanation is generated without Gemini response."
        )

    def generate_gemini_explanation(self, data: Dict) -> str:
        defects: List[Dict] = data.get("defects", [])
        risk_level = data.get("risk_level", "UNKNOWN")
        failure_probability = data.get("failure_probability", "N/A")
        status = data.get("status", "UNKNOWN")

        defect_lines = []
        for i, d in enumerate(defects, start=1):
            label = d.get("label", "unknown")
            conf = d.get("confidence", 0)
            bbox = d.get("bbox", [])
            defect_lines.append(f"{i}. type={label}, confidence={conf}, bbox={bbox}")

        defects_block = "\n".join(defect_lines) if defect_lines else "No defects detected."

        prompt = f"""
You are a senior PCB reliability engineer. Write a point-by-point technical explanation for this exact inspection.

Inspection data:
- Status: {status}
- Risk level: {risk_level}
- Failure probability: {failure_probability}%
- Detected defects:
{defects_block}

Output format requirements:
- Use exactly 5 numbered sections:
1) Root Cause Analysis
2) Operational Risk
3) Failure Prediction
4) Repair Plan
5) Validation and Release Decision
- Under each section, provide 2 to 4 short bullet-style lines using '-' prefix.
- Keep each line specific to listed defects and confidence values.
- Mention process contributors when relevant (placement, stencil/paste, reflow, contamination, AOI thresholding).
- Do not use markdown symbols like *, **, #, or backticks.
""".strip()

        if not self.enabled or self._client is None:
            return self._build_structured_fallback_explanation(data)

        last_exc = None
        for model_name in self.fallback_models:
            try:
                model = self._client.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = getattr(response, "text", "").strip()
                if text:
                    self.model_name = model_name
                    return self._to_plain_text(text) or DEFAULT_EXPLANATION
            except Exception as exc:
                last_exc = exc
                continue

        self.last_error = f"Gemini request failed: {last_exc}" if last_exc else "Gemini request failed."
        # Avoid repeated failing calls when key/model is invalid at runtime.
        self.enabled = False
        return self._build_structured_fallback_explanation(data)

    def generate_repair_suggestions(self, data: Dict, fallback_repairs: List[Dict]) -> List[Dict]:
        defects: List[Dict] = data.get("defects", [])
        risk_level = data.get("risk_level", "UNKNOWN")
        failure_probability = data.get("failure_probability", "N/A")
        status = data.get("status", "UNKNOWN")

        if not defects:
            return fallback_repairs

        if not self.enabled or self._client is None:
            return fallback_repairs

        defect_lines = []
        for i, d in enumerate(defects, start=1):
            label = d.get("label", "unknown")
            conf = d.get("confidence", 0)
            bbox = d.get("bbox", [])
            defect_lines.append(f"{i}. type={label}, confidence={conf}, bbox={bbox}")

        defects_block = "\n".join(defect_lines)

        prompt = f"""
You are a senior PCB manufacturing repair engineer.
Create practical repair suggestions for this exact inspection.

Inspection data:
- Status: {status}
- Risk level: {risk_level}
- Failure probability: {failure_probability}%
- Detected defects:
{defects_block}

Return ONLY valid JSON array. No markdown, no prose.
Each item must be:
{{
  "defect": "defect label",
  "suggestion": "specific repair action with process detail",
  "priority": "HIGH|MEDIUM|LOW",
  "validation": "how to verify repair success"
}}

Rules:
- Keep each suggestion concise and actionable.
- Mention manufacturing context when relevant (paste volume, reflow profile, alignment, contamination, solder bridge removal, continuity test).
- Suggestions must match listed defects and confidence.
""".strip()

        last_exc = None
        for model_name in self.fallback_models:
            try:
                model = self._client.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = getattr(response, "text", "").strip()
                parsed = self._extract_json_array(text)
                if not parsed:
                    continue
                normalized = self._normalize_repairs(parsed)
                if normalized:
                    self.model_name = model_name
                    return normalized
            except Exception as exc:
                last_exc = exc
                continue

        self.last_error = f"Gemini repair suggestion failed: {last_exc}" if last_exc else "Gemini repair suggestion failed."
        return fallback_repairs

    def generate_detection_assist(self, data: Dict) -> List[Dict]:
        defects: List[Dict] = data.get("defects", [])
        status = data.get("status", "UNKNOWN")
        model_mode = data.get("model_mode", "unknown")

        if not defects:
            return []
        if not self.enabled or self._client is None:
            return []

        defect_lines = []
        for idx, d in enumerate(defects):
            defect_lines.append(
                f"{idx}. label={d.get('label', 'unknown')}, "
                f"confidence={d.get('confidence', 0)}, "
                f"bbox={d.get('bbox', [])}, "
                f"model={d.get('model', 'unknown')}"
            )

        prompt = f"""
You are reviewing YOLO PCB defect detections for false-positive control.

Inspection status: {status}
Inference mode: {model_mode}
Detected defects:
{chr(10).join(defect_lines)}

Return ONLY valid JSON array (no markdown, no explanation text).
You must return at most one row per index listed above.
Each row format:
{{
  "index": 0,
  "ai_verdict": "CONFIRMED|SUSPECT_FALSE_POSITIVE|UNCERTAIN",
  "ai_confidence": 0.0,
  "reason": "short technical reason"
}}

Rules:
- Do not invent new detections.
- Be conservative: if uncertain, return UNCERTAIN.
- Use confidence and label plausibility to reduce false positives.
""".strip()

        last_exc = None
        for model_name in self.fallback_models:
            try:
                model = self._client.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = getattr(response, "text", "").strip()
                parsed = self._extract_json_array(text)
                if not parsed:
                    continue
                normalized = self._normalize_detection_audit(parsed, total_defects=len(defects))
                if normalized:
                    self.model_name = model_name
                    return normalized
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc:
            self.last_error = f"Gemini detection assist failed: {last_exc}"
            self.enabled = False
        return []
