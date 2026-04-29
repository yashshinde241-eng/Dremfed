"""
DermFed - tee_engine.py  (v3.0 - Groq edition)
TEE simulation layer + Groq LLaMA Vision VLM integration.

WHY GROQ (FREE TIER):
  - 30 requests/min, 14,400 requests/day - completely free
  - No credit card required
  - Native vision support via meta-llama/llama-4-scout-17b-16e-instruct
  - ~0.5-1 second response time (LPU hardware)
  - No rate limit issues on normal usage

TEE SIMULATION:
  1. PII scrubbing  - strip ALL EXIF before VLM touch
  2. Data isolation - VLM receives ONLY sanitised GradCAM overlay + text
  3. Audit trail    - every call logged with hash only (never raw data)
  4. Context fence  - system prompt bans personal info retention
  5. Minimal disclosure - GradCAM overlay only, not raw patient image

SETUP:
  1. Get free API key: https://console.groq.com  (no credit card)
  2. Set env variable:
       Windows CMD:        set GROQ_API_KEY=your_key_here
       Windows PowerShell: $env:GROQ_API_KEY="your_key_here"
       Or create a .env file in project root (loaded automatically)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from groq import Groq
from PIL import Image

# ── Config --------------------------------------------------------------------
GROQ_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
AUDIT_LOG_PATH = Path("results/tee_audit.jsonl")
TEE_TIMEOUT_SEC = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DermFed.TEE")


# ── .env loader (optional convenience) ---------------------------------------
def _load_dotenv() -> None:
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()
# Re-read after .env load
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ── Audit logger --------------------------------------------------------------
class AuditLogger:
    def __init__(self, path: Path = AUDIT_LOG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, event: str, payload: dict) -> None:
        record = {
            "event_id":  str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event":     event,
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


_audit = AuditLogger()


# ── TEE image helpers ---------------------------------------------------------
def scrub_image(image: Image.Image) -> Image.Image:
    """Strip ALL EXIF metadata - GPS, device IDs, patient timestamps."""
    clean = Image.new("RGB", image.size)
    clean.paste(image.convert("RGB"))
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).copy()


def image_to_b64(image: Image.Image, max_size: int = 512) -> str:
    """Resize to max_size on longest side + base64-encode as JPEG."""
    w, h  = image.size
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def input_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


# ── API key check -------------------------------------------------------------
def check_groq_available() -> tuple[bool, str]:
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_key_here":
        return False, (
            "GROQ_API_KEY not set.\n"
            "Get a free key at https://console.groq.com\n"
            "Then set it: $env:GROQ_API_KEY=\"your_key\" (PowerShell)"
        )
    try:
        client = Groq(api_key=key)
        client.models.list()
        return True, f"Groq API ready - model: {GROQ_MODEL}"
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return False, "Invalid GROQ_API_KEY. Check your key at console.groq.com."
        if "connection" in err.lower():
            return False, "No internet connection - cannot reach Groq API."
        return False, f"Groq API error: {e}"


# ── Prompt builder ------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a dermatology AI assistant explaining skin lesion classifications "
    "to medical professionals. You are operating in a privacy-enforced TEE context. "
    "RULES: (1) Do NOT reference any patient identity or personal information - "
    "the image is fully anonymised. (2) Focus only on visual lesion characteristics. "
    "(3) Explain WHY the CNN made its prediction, not make an independent diagnosis."
)


def build_vlm_prompt(pred_class_name, confidence, region_desc, all_probs, class_names):
    top3     = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)[:3]
    top3_str = "\n".join(f"  {class_names[i]}: {p*100:.1f}%" for i, p in top3)
    conf_pct = f"{confidence*100:.1f}%"
    return (
        f"A MobileNetV2 CNN trained via Federated Learning on HAM10000 predicted:\n"
        f"Diagnosis: {pred_class_name}\n"
        f"Confidence: {conf_pct}\n\n"
        f"Grad-CAM attention was concentrated in {region_desc}.\n"
        f"Top-3 class probabilities:\n{top3_str}\n\n"
        f"You are viewing the Grad-CAM overlay blended on the anonymised lesion image.\n"
        f"Red/yellow = high model attention. Blue = low attention.\n\n"
        f"Provide a structured explanation with exactly these 4 sections:\n\n"
        f"**1. Visual Evidence** (2-3 sentences)\n"
        f"What visual features in highlighted regions are consistent with {pred_class_name}?\n\n"
        f"**2. Why This Prediction** (2-3 sentences)\n"
        f"What dermoscopic criteria for {pred_class_name} does the model appear to detect?\n\n"
        f"**3. Differential Diagnosis** (1-2 sentences)\n"
        f"What visual ambiguity explains the 2nd and 3rd ranked classes?\n\n"
        f"**4. Confidence Assessment** (1 sentence)\n"
        f"Is {conf_pct} confidence clinically appropriate here?\n\n"
        f"Keep total response under 300 words. Be precise and clinical."
    )


# ── Main engine --------------------------------------------------------------
class ConfidentialInferenceEngine:
    """
    TEE-simulated inference engine using Gemini 1.5 Flash (free tier).

    Privacy pipeline:
      raw image -> EXIF scrub -> resize GradCAM overlay -> base64
      -> Gemini API (HTTPS, no storage) -> structured explanation
      -> audit log (hash only, never raw pixels)
    """

    def __init__(self) -> None:
        self.available, self.status_msg = check_groq_available()
        logger.info(f"[TEE] {self.status_msg}")

    def refresh_status(self) -> tuple[bool, str]:
        self.available, self.status_msg = check_groq_available()
        return self.available, self.status_msg

    def generate_explanation(
        self,
        original_image  : Image.Image,
        overlay_image   : Image.Image,
        pred_class_idx  : int,
        pred_class_name : str,
        confidence      : float,
        all_probs       : list[float],
        class_names     : list[str],
        region_desc     : str,
    ) -> dict[str, Any]:
        t_start  = time.time()
        audit_id = str(uuid.uuid4())[:8]
        img_hash = input_hash(original_image)

        _audit.log("inference_request", {
            "audit_id":     audit_id,
            "image_hash":   img_hash,
            "pred_class":   pred_class_name,
            "confidence":   round(confidence, 4),
            "pii_scrubbed": True,
            "vlm_sees":     "gradcam_overlay_only",
            "provider":     "gemini_flash",
        })

        # TEE Step 1: Scrub PII from overlay image
        clean_overlay = scrub_image(overlay_image)
        b64_image     = image_to_b64(clean_overlay, max_size=512)

        # TEE Step 2: Build prompt
        prompt = build_vlm_prompt(
            pred_class_name, confidence, region_desc, all_probs, class_names
        )

        if not self.available:
            _audit.log("inference_skipped", {"audit_id": audit_id, "reason": "api_unavailable"})
            return {
                "success":     False,
                "explanation": self.status_msg,
                "audit_id":    audit_id,
                "latency_ms":  0,
                "tee_status":  self._tee_status(img_hash),
            }

        # TEE Step 3: Call Groq API
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=700,
            )

            explanation = response.choices[0].message.content.strip()
            latency_ms  = int((time.time() - t_start) * 1000)

            _audit.log("inference_complete", {
                "audit_id":   audit_id,
                "latency_ms": latency_ms,
                "words":      len(explanation.split()),
                "provider":   "groq",
            })

            return {
                "success":     True,
                "explanation": explanation,
                "audit_id":    audit_id,
                "latency_ms":  latency_ms,
                "tee_status":  self._tee_status(img_hash),
            }

        except Exception as e:
            msg = f"API error: {e}"
            logger.error(f"[TEE] {msg}")
            _audit.log("inference_error", {"audit_id": audit_id, "error": str(e)})
            return {"success": False, "explanation": msg, "audit_id": audit_id,
                    "latency_ms": 0, "tee_status": self._tee_status(img_hash)}

    def _tee_status(self, img_hash: str) -> dict:
        return {
            "pii_scrubbed":       True,
            "exif_stripped":      True,
            "vlm_input":          "gradcam_overlay_only",
            "vlm_model":          GROQ_MODEL,
            "vlm_provider":       "Groq (free tier)",
            "data_leaves_device": True,
            "tee_note":           "Image sent over HTTPS to Groq API. No persistent storage per Groq policy.",
            "image_hash":         img_hash,
            "audit_log":          str(AUDIT_LOG_PATH),
        }


_engine: ConfidentialInferenceEngine | None = None


def get_engine() -> ConfidentialInferenceEngine:
    global _engine
    if _engine is None:
        _engine = ConfidentialInferenceEngine()
    return _engine
