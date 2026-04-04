import os
from typing import Dict, List


DEFAULT_EXPLANATION = (
    "Gemini response unavailable. Review defect confidence, perform electrical validation, "
    "and apply recommended repairs before production release."
)


class GeminiEngine:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
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
            self.last_error = "GEMINI_API_KEY is missing."

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
You are a senior PCB reliability engineer. Generate a precise, non-generic technical analysis for this exact inspection.

Inspection data:
- Status: {status}
- Risk level: {risk_level}
- Failure probability: {failure_probability}%
- Detected defects:\n{defects_block}

Respond with these 5 sections and make each section specific to the provided defects:
1) Root Cause Analysis
2) Operational Risks
3) Failure Prediction
4) Repair Explanation
5) Engineering Insights

Rules:
- Tie each statement to given defect types and confidence values.
- Mention likely manufacturing process contributors when relevant (placement, soldering, reflow, contamination, AOI calibration).
- Keep the response concise but technically actionable.
""".strip()

        if not self.enabled or self._client is None:
            reason = self.last_error or "Gemini is disabled."
            return f"{DEFAULT_EXPLANATION} Reason: {reason}"

        try:
            model = self._client.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", "").strip()
            return text or DEFAULT_EXPLANATION
        except Exception as exc:
            self.last_error = f"Gemini request failed: {exc}"
            return f"{DEFAULT_EXPLANATION} Reason: {self.last_error}"
