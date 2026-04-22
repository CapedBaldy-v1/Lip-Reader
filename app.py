"""
Swin-VALLR Demo Application
============================
Real-time lip reading demo with "Anti-Gravity" futuristic UI.

Features:
- WebRTC webcam capture
- Real-time lip detection and transcription
- Glassmorphism UI theme
- Floating subtitle display
"""

import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
import torch
from tqdm import tqdm

from logging_utils import setup_logging, log_system_info, log_exception
from artifact_utils import get_useful_dir, save_json, save_montage, save_text

from nlp_utils import get_translation_options, translate_text, summarize_text

LOGGER = setup_logging("app")

# Streamlit imports
import streamlit as st

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("[App] streamlit-webrtc not available - using file upload mode")
    LOGGER.warning("streamlit-webrtc not available; using file upload mode.")

# Local imports
from backend_manager import get_device, get_backend, get_backend_name, print_device_info
from model_architecture import SwinVALLR, SwinConfig, create_model

try:
    from data_pipeline import MediaPipeProcessor
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    LOGGER.warning("MediaPipe import failed; lip extraction disabled.")


# =============================================================================
# Constants
# =============================================================================

ROI_SIZE = 96
TARGET_FRAMES = 60
MIN_FRAMES_FOR_INFERENCE = 60
DEFAULT_INFERENCE_INTERVAL = 1
MONTAGE_FRAMES = 60

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


# =============================================================================
# Anti-Gravity CSS Theme
# =============================================================================

ANTI_GRAVITY_CSS = """
<style>
/* Global Styles */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Rajdhani:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #00f5ff;
    --secondary-color: #b026ff;
    --accent-color: #ff2d95;
    --bg-dark: #0a0a1a;
    --bg-gradient-start: #1a1a2e;
    --bg-gradient-end: #0f0f1a;
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Main app background */
.stApp {
    background: radial-gradient(ellipse at top, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 50%, var(--bg-dark) 100%);
    font-family: 'Rajdhani', sans-serif;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: var(--primary-color) !important;
    text-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
}

/* Floating card effect */
.floating-card {
    backdrop-filter: blur(16px);
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 
        0 8px 32px 0 rgba(0, 0, 0, 0.37),
        inset 0 0 32px rgba(255, 255, 255, 0.025);
    animation: float 6s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

.floating-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(0, 245, 255, 0.1),
        transparent
    );
    animation: shimmer 8s linear infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Phoneme display */
.phoneme-box {
    backdrop-filter: blur(10px);
    background: linear-gradient(135deg, rgba(176, 38, 255, 0.2), rgba(0, 245, 255, 0.2));
    border: 1px solid var(--secondary-color);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    font-family: 'Orbitron', monospace;
    color: var(--text-primary);
    letter-spacing: 2px;
    text-align: center;
    font-size: 1.2em;
    box-shadow: 0 0 20px rgba(176, 38, 255, 0.3);
}

/* Refined text display */
.refined-text-box {
    backdrop-filter: blur(10px);
    background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(255, 45, 149, 0.2));
    border: 1px solid var(--primary-color);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    color: var(--text-primary);
    font-size: 1.5em;
    font-weight: 500;
    text-align: center;
    box-shadow: 
        0 0 30px rgba(0, 245, 255, 0.4),
        inset 0 0 20px rgba(0, 245, 255, 0.1);
    animation: pulse-glow 3s ease-in-out infinite;
}

/* YouTube-style captions */
.caption-box {
    margin-top: 16px;
    background: rgba(0, 0, 0, 0.75);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
}

.caption-text {
    color: #ffffff;
    font-size: 1.3em;
    font-weight: 600;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.8);
    margin-bottom: 6px;
}

.caption-translation {
    color: rgba(255, 255, 255, 0.75);
    font-size: 1.05em;
    font-weight: 500;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 30px rgba(0, 245, 255, 0.4); }
    50% { box-shadow: 0 0 50px rgba(0, 245, 255, 0.6); }
}

/* Status indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 500;
}

.status-online {
    background: rgba(0, 255, 136, 0.2);
    color: #00ff88;
    border: 1px solid rgba(0, 255, 136, 0.3);
}

.status-offline {
    background: rgba(255, 68, 68, 0.2);
    color: #ff4444;
    border: 1px solid rgba(255, 68, 68, 0.3);
}

/* Pulsing dot */
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

.pulse-dot.online {
    background: #00ff88;
    box-shadow: 0 0 10px #00ff88;
}

.pulse-dot.offline {
    background: #ff4444;
    box-shadow: 0 0 10px #ff4444;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* Video container styling */
.video-container {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 
        0 0 40px rgba(0, 245, 255, 0.3),
        0 0 80px rgba(176, 38, 255, 0.2);
    border: 2px solid var(--glass-border);
}

/* Metrics display */
.metric-card {
    backdrop-filter: blur(10px);
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2em;
    color: var(--primary-color);
    text-shadow: 0 0 10px var(--primary-color);
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9em;
    margin-top: 4px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-gradient-start) 0%, var(--bg-dark) 100%);
    border-right: 1px solid var(--glass-border);
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    font-size: 1em !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(176, 38, 255, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 245, 255, 0.5) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--secondary-color), var(--primary-color));
    border-radius: 4px;
}
</style>
"""


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        'model': None,
        'lip_processor': None,
        'phoneme_buffer': [],
        'text_buffer': "",
        'translation_buffer': "",
        'speaker_buffer': "Speaker 1",
        'frame_buffer': [],
        'inference_running': False,
        'last_inference_time': 0,
        'fps': 0,
        'transcript_lines': [],
        'summary_text': "",
        'webrtc_active': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Model Loading
# =============================================================================

@st.cache_resource
def load_model(
    checkpoint_path: Optional[str] = None,
    load_refiner: bool = True,
    use_checkpoint: bool = True,
    allow_random_output: bool = False
):
    """Load Swin-VALLR model with caching."""
    print("[App] Loading model...")
    LOGGER.info(
        "Loading model (checkpoint=%s, load_refiner=%s, use_checkpoint=%s).",
        checkpoint_path,
        load_refiner,
        use_checkpoint
    )

    checkpoint = None
    model_config = None
    if use_checkpoint and checkpoint_path and Path(checkpoint_path).exists():
        map_location = "cpu" if get_backend() == "directml" else get_device()
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        raw_config = checkpoint.get("model_config")
        if raw_config:
            try:
                model_config = SwinConfig(**raw_config)
                LOGGER.info("Loaded model config from checkpoint.")
            except Exception:
                log_exception(LOGGER, "Failed to parse model_config from checkpoint.")

    model = create_model(load_refiner=False, config=model_config)

    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            log_exception(LOGGER, "Strict checkpoint load failed; retrying with strict=False.")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"[App] Loaded checkpoint: {checkpoint_path}")
        LOGGER.info("Loaded checkpoint: %s", checkpoint_path)

    if load_refiner:
        try:
            adapter_path = os.environ.get("SWIN_VALLR_REFINER_ADAPTER")
            model.load_refiner(adapter_path=adapter_path)
        except Exception as e:
            log_exception(LOGGER, "Failed to load linguistic refiner.")
            print(f"[App] Failed to load refiner: {e}")

    model.eval()
    model._has_checkpoint = checkpoint is not None
    model._checkpoint_path = str(checkpoint_path) if checkpoint is not None else None
    model._use_checkpoint = bool(use_checkpoint)
    model._allow_random_output = bool(allow_random_output)
    return model


@st.cache_resource
def load_lip_processor(
    max_faces: int,
    align_mouth: bool,
    profile_aware: bool,
    profile_expand: float,
    profile_yaw_scale: float
):
    """Load MediaPipe processor with caching."""
    if not MEDIAPIPE_AVAILABLE:
        return None
    LOGGER.info(
        "Loading MediaPipeProcessor (max_faces=%s, align_mouth=%s, profile_aware=%s).",
        max_faces,
        align_mouth,
        profile_aware
    )
    try:
        return MediaPipeProcessor(
            roi_size=ROI_SIZE,
            stabilize=True,
            max_faces=max_faces,
            align_mouth=align_mouth,
            profile_aware=profile_aware,
            profile_expand=profile_expand,
            profile_yaw_scale=profile_yaw_scale
        )
    except Exception as exc:
        LOGGER.warning("MediaPipeProcessor unavailable; falling back to center crop. %s", exc)
        return None


# =============================================================================
# Video Processing
# =============================================================================

class VideoProcessor:
    """WebRTC video processor callback."""

    def __init__(self):
        self.frame_count = 0
        self.inference_interval = DEFAULT_INFERENCE_INTERVAL
        self.frame_buffer = []
        self.result_queue = queue.Queue(maxsize=10)
        self.model = None
        self.lip_processor = None
        self.device = None
        self.use_nbest = False
        self.nbest = 3
        self.beam_width = 10
        self.multi_speaker = False
        self.live_inference = False
        self.live_inference_interval_sec = _env_float(
            "SWIN_VALLR_LIVE_INFER_SEC", 1.0, min_value=0.1
        )
        self._last_inference_ts = 0.0
        self._inference_lock = threading.Lock()
        self.latest_caption = ""
        self.latest_speaker = ""
        self._last_error = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame."""
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.inference_interval == 0:
            self.frame_buffer.append(img.copy())
            if len(self.frame_buffer) > TARGET_FRAMES:
                self.frame_buffer.pop(0)
            self._maybe_start_inference()

        img = self._draw_overlay(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _maybe_start_inference(self) -> None:
        if not self.live_inference:
            return
        if self.model is None or self.device is None:
            return
        if len(self.frame_buffer) < MIN_FRAMES_FOR_INFERENCE:
            return
        now = time.time()
        if (now - self._last_inference_ts) < self.live_inference_interval_sec:
            return
        if self._inference_lock.locked():
            return
        frames = list(self.frame_buffer)
        self._last_inference_ts = now
        thread = threading.Thread(
            target=self._run_inference,
            args=(frames,),
            daemon=True
        )
        thread.start()

    def _run_inference(self, frames: List[np.ndarray]) -> None:
        with self._inference_lock:
            try:
                phonemes, text, speaker_id, decoding = process_frames_for_inference(
                    frames,
                    self.lip_processor,
                    self.model,
                    self.device,
                    use_nbest=self.use_nbest,
                    nbest=self.nbest,
                    beam_width=self.beam_width,
                    multi_speaker=self.multi_speaker
                )
            except Exception as exc:
                self._last_error = str(exc)
                return

            placeholder = bool(decoding.get("untrained_guard")) if decoding else False
            if placeholder and not text:
                text = (
                    "Untrained model: output suppressed. "
                    "Load a checkpoint or set SWIN_VALLR_ALLOW_RANDOM_OUTPUT=true."
                )
            self.latest_caption = text or ""
            self.latest_speaker = speaker_id or ""
            try:
                self.result_queue.put_nowait({
                    "phonemes": phonemes,
                    "text": text,
                    "speaker": speaker_id or "Speaker 1",
                    "placeholder": placeholder
                })
            except queue.Full:
                pass

    def _draw_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw detection overlay on frame."""
        height, width = img.shape[:2]
        bracket_length = 30
        bracket_thickness = 2
        color = (0, 245, 255)

        # Top-left
        cv2.line(img, (10, 10), (10 + bracket_length, 10), color, bracket_thickness)
        cv2.line(img, (10, 10), (10, 10 + bracket_length), color, bracket_thickness)

        # Top-right
        cv2.line(img, (width - 10, 10), (width - 10 - bracket_length, 10), color, bracket_thickness)
        cv2.line(img, (width - 10, 10), (width - 10, 10 + bracket_length), color, bracket_thickness)

        # Bottom-left
        cv2.line(img, (10, height - 10), (10 + bracket_length, height - 10), color, bracket_thickness)
        cv2.line(img, (10, height - 10), (10, height - 10 - bracket_length), color, bracket_thickness)

        # Bottom-right
        cv2.line(img, (width - 10, height - 10), (width - 10 - bracket_length, height - 10), color, bracket_thickness)
        cv2.line(img, (width - 10, height - 10), (width - 10, height - 10 - bracket_length), color, bracket_thickness)

        cv2.putText(
            img, "SWIN-VALLR", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        if self.live_inference:
            caption = self.latest_caption or "Listening..."
            speaker = self.latest_speaker.strip()
            if speaker:
                caption = f"{speaker}: {caption}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            max_width = width - 20
            lines = _wrap_caption_lines(caption, max_width, font, scale, thickness)
            if lines:
                line_height = cv2.getTextSize("Ay", font, scale, thickness)[0][1] + 6
                box_height = line_height * len(lines) + 16
                overlay = img.copy()
                cv2.rectangle(
                    overlay,
                    (0, height - box_height),
                    (width, height),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                y = height - box_height + line_height
                for line in lines:
                    cv2.putText(
                        img,
                        line,
                        (10, y),
                        font,
                        scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )
                    y += line_height

        return img


def _truncate_caption(text: str, max_width: int, font, scale: float, thickness: int) -> str:
    if not text:
        return ""
    width = cv2.getTextSize(text, font, scale, thickness)[0][0]
    if width <= max_width:
        return text
    trimmed = text
    while trimmed and cv2.getTextSize(trimmed + "...", font, scale, thickness)[0][0] > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + "...") if trimmed else "..."


def _wrap_caption_lines(
    text: str,
    max_width: int,
    font,
    scale: float,
    thickness: int,
    max_lines: int = 2
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current = ""
    truncated = False
    idx = -1
    for idx, word in enumerate(words):
        candidate = word if not current else f"{current} {word}"
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(_truncate_caption(word, max_width, font, scale, thickness))
            current = ""
        if len(lines) >= max_lines:
            truncated = True
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if truncated or idx < len(words) - 1:
        lines[-1] = _truncate_caption(lines[-1], max_width, font, scale, thickness)
    return lines


def _center_crop_roi(frame: np.ndarray) -> np.ndarray:
    """Fallback crop when MediaPipe is not available."""
    height, width = frame.shape[:2]
    size = min(height, width) // 2
    center_y, center_x = height // 2, width // 2
    roi = frame[
        center_y - size // 2:center_y + size // 2,
        center_x - size // 2:center_x + size // 2
    ]
    return cv2.resize(roi, (ROI_SIZE, ROI_SIZE))


class FallbackLipTracker:
    """Lightweight face-based mouth ROI tracker for MediaPipe-less runs."""

    def __init__(self):
        self.detect_interval = _env_int("SWIN_VALLR_FALLBACK_DETECT_INTERVAL", 5, min_value=1)
        self.min_face_size = _env_int("SWIN_VALLR_FALLBACK_FACE_MIN", 60, min_value=20)
        self.smooth = _env_float("SWIN_VALLR_FALLBACK_SMOOTH", 0.6, min_value=0.0)
        self._frame_idx = 0
        self._face_box = None
        self._detector = self._load_detector()

    def _load_detector(self):
        haar_root = getattr(cv2.data, "haarcascades", "")
        if not haar_root:
            return None
        model_path = os.path.join(haar_root, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(model_path)
        if detector.empty():
            return None
        return detector

    def _detect_face(self, frame: np.ndarray):
        if self._detector is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 0.5
        small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        faces = self._detector.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(self.min_face_size * scale), int(self.min_face_size * scale))
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

    def _smooth_box(self, box):
        if self._face_box is None:
            self._face_box = box
            return
        alpha = max(0.0, min(self.smooth, 1.0))
        x0, y0, w0, h0 = self._face_box
        x1, y1, w1, h1 = box
        self._face_box = (
            int(x0 + alpha * (x1 - x0)),
            int(y0 + alpha * (y1 - y0)),
            int(w0 + alpha * (w1 - w0)),
            int(h0 + alpha * (h1 - h0))
        )

    def _mouth_roi(self, frame: np.ndarray):
        if self._face_box is None:
            return None
        height, width = frame.shape[:2]
        x, y, w, h = self._face_box
        mouth_y = y + int(h * 0.6)
        mouth_h = int(h * 0.35)
        mouth_x = x + int(w * 0.15)
        mouth_w = int(w * 0.7)
        x1 = max(0, mouth_x)
        y1 = max(0, mouth_y)
        x2 = min(width, mouth_x + mouth_w)
        y2 = min(height, mouth_y + mouth_h)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return cv2.resize(roi, (ROI_SIZE, ROI_SIZE))

    def extract(self, frame: np.ndarray) -> np.ndarray:
        self._frame_idx += 1
        if self._frame_idx % self.detect_interval == 0 or self._face_box is None:
            detected = self._detect_face(frame)
            if detected is not None:
                self._smooth_box(detected)
        roi = self._mouth_roi(frame)
        if roi is None:
            roi = _center_crop_roi(frame)
        return roi


def _extract_lip_rois(
    frames: List[np.ndarray],
    lip_processor: Optional['MediaPipeProcessor'],
    multi_speaker: bool = False
) -> Tuple[List[np.ndarray], str]:
    """Extract lip ROIs from raw frames."""
    if multi_speaker and lip_processor is not None and getattr(lip_processor, "max_faces", 1) > 1:
        return _extract_lip_rois_multi(frames, lip_processor)

    rois: List[np.ndarray] = []

    fallback_tracker = FallbackLipTracker() if lip_processor is None else None
    for frame in tqdm(frames, desc="Extracting lip ROIs", unit="frame", leave=False):
        if lip_processor is None:
            roi = fallback_tracker.extract(frame) if fallback_tracker else _center_crop_roi(frame)
            rois.append(roi)
            continue

        roi = lip_processor.extract_lip_roi(frame)
        if roi is not None:
            rois.append(roi)
        elif rois:
            rois.append(rois[-1].copy())
        else:
            rois.append(_center_crop_roi(frame))

    return rois, "Speaker 1"


def _extract_lip_rois_multi(
    frames: List[np.ndarray],
    lip_processor: 'MediaPipeProcessor'
) -> Tuple[List[np.ndarray], str]:
    """Extract lip ROIs for the most active speaker in multi-face frames."""
    if lip_processor is None:
        return _extract_lip_rois(frames, None, multi_speaker=False)

    if not hasattr(lip_processor, "extract_face_observations") or not hasattr(lip_processor, "extract_lip_roi_from_landmarks"):
        LOGGER.warning("Multi-speaker extraction unavailable; falling back.")
        return _extract_lip_rois(frames, lip_processor, multi_speaker=False)

    rois: List[np.ndarray] = []
    speaker_votes: List[int] = []
    tracks: Dict[int, Tuple[float, float]] = {}
    next_track_id = 1
    prev_centers: List[Tuple[float, float]] = []
    prev_openness: List[float] = []
    max_distance = float(ROI_SIZE) * 1.5

    for frame in tqdm(frames, desc="Selecting active speaker", unit="frame", leave=False):
        try:
            observations = lip_processor.extract_face_observations(frame)
        except Exception:
            log_exception(LOGGER, "Multi-speaker face observation failed.")
            return _extract_lip_rois(frames, lip_processor, multi_speaker=False)

        if not observations:
            if rois:
                rois.append(rois[-1].copy())
            else:
                rois.append(_center_crop_roi(frame))
            continue

        centers: List[Tuple[float, float]] = []
        openness_vals: List[float] = []
        track_ids: List[int] = []
        for metrics, _ in observations:
            center = metrics.center
            centers.append(center)
            openness_vals.append(metrics.openness)

            track_id = None
            if tracks:
                best_id, best_dist = min(
                    ((tid, np.linalg.norm(np.array(center) - np.array(prev_center)))
                     for tid, prev_center in tracks.items()),
                    key=lambda item: item[1]
                )
                if best_dist <= max_distance:
                    track_id = best_id

            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
            track_ids.append(track_id)

        for center, track_id in zip(centers, track_ids):
            tracks[track_id] = center

        scores: List[float] = []
        for metrics in (obs[0] for obs in observations):
            motion = 0.0
            if prev_centers:
                distances = [
                    np.linalg.norm(np.array(metrics.center) - np.array(prev_center))
                    for prev_center in prev_centers
                ]
                prev_idx = int(np.argmin(distances))
                motion = abs(metrics.openness - prev_openness[prev_idx])
            size_score = metrics.size / float(max(1.0, ROI_SIZE))
            score = metrics.openness + (0.75 * motion) + (0.5 * size_score)
            scores.append(score)

        best_idx = int(np.argmax(scores)) if scores else 0
        _, landmarks = observations[best_idx]
        roi = lip_processor.extract_lip_roi_from_landmarks(frame, landmarks)
        if roi is None:
            roi = rois[-1].copy() if rois else _center_crop_roi(frame)
        rois.append(roi)

        if track_ids:
            speaker_votes.append(track_ids[best_idx])

        prev_centers = centers
        prev_openness = openness_vals

    speaker_id = "Speaker 1"
    if speaker_votes:
        most_common = Counter(speaker_votes).most_common(1)[0][0]
        speaker_id = f"Speaker {most_common}"

    return rois, speaker_id


def _pad_or_trim_rois(rois: List[np.ndarray]) -> List[np.ndarray]:
    """Pad or trim to the exact clip length."""
    while len(rois) < TARGET_FRAMES:
        if rois:
            rois.append(rois[-1].copy())
        else:
            rois.append(np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8))

    return rois[:TARGET_FRAMES]


def _prepare_video_tensor(rois: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Normalize and reshape ROIs into a model input tensor."""
    video = np.stack(rois, axis=0).astype(np.float32) / 255.0
    video = (video - IMAGENET_MEAN) / IMAGENET_STD
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()
    return video.to(device)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    LOGGER.warning("Invalid %s=%s; using default=%s.", name, value, default)
    return default


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        LOGGER.warning("Invalid %s=%s; using default=%s.", name, value, default)
        return default
    return max(min_value, parsed)


def _env_float(name: str, default: float, min_value: Optional[float] = None) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        LOGGER.warning("Invalid %s=%s; using default=%s.", name, value, default)
        return default
    if min_value is not None:
        return max(min_value, parsed)
    return parsed


def _allow_random_output() -> bool:
    return _env_bool("SWIN_VALLR_ALLOW_RANDOM_OUTPUT", False)


def _effective_allow_random_output(model: SwinVALLR) -> bool:
    env_value = os.getenv("SWIN_VALLR_ALLOW_RANDOM_OUTPUT")
    if env_value is not None and env_value.strip():
        return _allow_random_output()
    if getattr(model, "_allow_random_output", False):
        return True
    if not getattr(model, "_use_checkpoint", True):
        return True
    return False


def _untrained_guard(logits: torch.Tensor, model: SwinVALLR) -> Optional[float]:
    """Return average confidence when suppressing untrained outputs."""
    if getattr(model, "_has_checkpoint", False):
        return None
    if _effective_allow_random_output(model):
        return None
    if logits.ndim != 3:
        return None
    max_probs = logits.exp().max(dim=-1).values
    if max_probs.numel() == 0:
        return 0.0
    avg_conf = float(max_probs.mean().item())
    if avg_conf < 0.2:
        return avg_conf
    return None


def _run_model_inference(
    model: SwinVALLR,
    video: torch.Tensor,
    use_nbest: bool = False,
    nbest: int = 5,
    beam_width: int = 10
) -> Tuple[List[str], str, Optional[object], Optional[Dict[str, object]]]:
    """Run the model and decode phonemes + refined text."""
    language_hint = os.getenv("SWIN_VALLR_LANGUAGE_HINT")
    if language_hint:
        language_hint = language_hint.strip() or None

    refiner_mode = os.getenv("SWIN_VALLR_REFINER_MODE", "phoneme").strip().lower()
    if refiner_mode not in {"phoneme", "roman", "hybrid"}:
        LOGGER.warning("Invalid SWIN_VALLR_REFINER_MODE=%s; using phoneme.", refiner_mode)
        refiner_mode = "phoneme"

    with torch.no_grad():
        logits = model(video)
        guard_conf = _untrained_guard(logits, model)
        if guard_conf is not None:
            decoding_detail = {
                "untrained_guard": True,
                "avg_confidence": guard_conf,
                "allow_random_output": _effective_allow_random_output(model)
            }
            return [], "", None, decoding_detail

        if use_nbest:
            use_dual_hyp = _env_bool("SWIN_VALLR_DUAL_HYP", True)
            smooth_kernel = _env_int("SWIN_VALLR_DUAL_HYP_SMOOTH_KERNEL", 3, min_value=1)
            chunk_frames = _env_int("SWIN_VALLR_DUAL_HYP_CHUNK_FRAMES", 10, min_value=1)

            if use_dual_hyp:
                dual = model.decode_ctc_dual_hypotheses(
                    logits,
                    beam_width=beam_width,
                    nbest=nbest,
                    smooth_kernel=smooth_kernel,
                    chunk_frames=chunk_frames
                )[0]
                primary = dual.get("primary", [])
                secondary = dual.get("secondary", [])
                primary_candidates = [item["phonemes"] for item in primary]
                secondary_candidates = [item["phonemes"] for item in secondary]
                primary_scores = [float(item["log_prob"]) for item in primary]
                secondary_scores = [float(item["log_prob"]) for item in secondary]

                if model.refiner is not None:
                    text = model.refiner.refine_dual_hypotheses(
                        primary_candidates,
                        secondary_candidates,
                        primary_scores=primary_scores,
                        secondary_scores=secondary_scores,
                        primary_reliability=dual.get("primary_reliability"),
                        secondary_reliability=dual.get("secondary_reliability"),
                        language_hint=language_hint,
                        mode=refiner_mode
                    )
                else:
                    combined_candidates = primary_candidates + secondary_candidates
                    combined_scores = primary_scores + secondary_scores
                    if combined_candidates:
                        if combined_scores:
                            best_idx = max(range(len(combined_scores)), key=lambda i: combined_scores[i])
                        else:
                            best_idx = 0
                        text = " ".join(combined_candidates[best_idx])
                    else:
                        text = ""

                phonemes = primary_candidates[0] if primary_candidates else (
                    secondary_candidates[0] if secondary_candidates else []
                )
                decoding_detail = {
                    "dual_hyp": True,
                    "dual_hyp_smooth_kernel": smooth_kernel,
                    "dual_hyp_chunk_frames": chunk_frames,
                    "dual_hyp_primary_reliability": dual.get("primary_reliability"),
                    "dual_hyp_secondary_reliability": dual.get("secondary_reliability")
                }
                return phonemes, text, dual, decoding_detail

            hypotheses = model.decode_ctc_nbest(
                logits,
                beam_width=beam_width,
                nbest=nbest
            )[0]
            candidates = [item["phonemes"] for item in hypotheses]
            scores = [float(item["log_prob"]) for item in hypotheses]

            if model.refiner is not None:
                text = model.refiner.refine_candidates(
                    candidates,
                    scores,
                    language_hint=language_hint,
                    mode=refiner_mode
                )
            else:
                best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
                text = ' '.join(candidates[best_idx]) if candidates else ""

            phonemes = candidates[0] if candidates else []
            decoding_detail = {"dual_hyp": False}
            return phonemes, text, hypotheses, decoding_detail

        use_uncertainty_prompt = _env_bool("SWIN_VALLR_UNCERTAINTY_PROMPT", True)
        alt_topk = _env_int("SWIN_VALLR_ALT_TOPK", 3, min_value=2)
        alt_max = _env_int("SWIN_VALLR_ALT_MAX", 2, min_value=1)
        accent_aware = _env_bool("SWIN_VALLR_ACCENT_AWARE", True)
        accent_max = _env_int("SWIN_VALLR_ACCENT_MAX", 2, min_value=1)
        if language_hint and language_hint.lower() not in {"english", "en"}:
            accent_aware = False

        if model.refiner is not None and use_uncertainty_prompt:
            alt_bundle = model.decode_ctc_with_alternatives(
                logits,
                top_k=alt_topk,
                max_alternatives=alt_max,
                accent_aware=accent_aware,
                accent_max_alternatives=accent_max
            )[0]
            phonemes = alt_bundle.get("phonemes", [])
            alternatives = alt_bundle.get("alternatives", [])
            confidences = alt_bundle.get("confidences", [])
            text = model.refiner.refine_phonemes_with_alternatives(
                phonemes,
                alternatives=alternatives,
                confidences=confidences,
                language_hint=language_hint,
                mode=refiner_mode
            ) if phonemes else ""
            decoding_detail = {
                "uncertainty_prompt": True,
                "alt_topk": alt_topk,
                "alt_max": alt_max
            }
            return phonemes, text, alt_bundle, decoding_detail

        phonemes = model.decode_ctc(logits)[0]
        if model.refiner is not None:
            text = model.refiner.refine_phonemes(
                phonemes,
                language_hint=language_hint,
                mode=refiner_mode
            )
        else:
            text = ' '.join(phonemes)

    return phonemes, text, None, None


def _save_inference_artifacts(
    frames: List[np.ndarray],
    rois: List[np.ndarray],
    phonemes: List[str],
    text: str,
    speaker_id: Optional[str] = None,
    hypotheses: Optional[object] = None,
    decoding: Optional[Dict[str, object]] = None
) -> None:
    """Persist inference artifacts for research."""
    try:
        root = get_useful_dir("app", "inference")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        save_montage(run_dir / "frames.png", frames[:min(len(frames), MONTAGE_FRAMES)], cols=10, pad=2)
        save_montage(run_dir / "rois.png", rois[:min(len(rois), MONTAGE_FRAMES)], cols=10, pad=2)
        payload = {
            "num_frames": len(frames),
            "num_rois": len(rois),
            "phonemes": phonemes,
            "text": text
        }
        if speaker_id:
            payload["speaker"] = speaker_id
        if hypotheses is not None:
            payload["hypotheses"] = hypotheses
        if decoding is not None:
            payload["decoding"] = decoding
        save_json(run_dir / "result.json", payload)
        save_text(run_dir / "phonemes.txt", " ".join(phonemes))
        save_text(run_dir / "text.txt", text)
    except Exception:
        log_exception(LOGGER, "Failed to save inference artifacts.")


def process_frames_for_inference(
    frames: List[np.ndarray],
    lip_processor: Optional['MediaPipeProcessor'],
    model: SwinVALLR,
    device: torch.device,
    use_nbest: bool = False,
    nbest: int = 5,
    beam_width: int = 10,
    multi_speaker: Optional[bool] = None
) -> Tuple[List[str], str, str, Dict[str, object]]:
    """
    Process buffered frames through the model.

    Returns:
        (phonemes, refined_text, speaker_id, decoding_meta)
    """
    if len(frames) < MIN_FRAMES_FOR_INFERENCE:
        LOGGER.warning("Not enough frames for inference: %s", len(frames))
        return [], "", "Speaker 1", {"reason": "not_enough_frames"}

    if multi_speaker is None:
        multi_speaker = _env_bool("SWIN_VALLR_MULTI_SPEAKER", False)
    rois, speaker_id = _extract_lip_rois(frames, lip_processor, multi_speaker=multi_speaker)
    rois = _pad_or_trim_rois(rois)
    video = _prepare_video_tensor(rois, device)

    phonemes, text, hypotheses, decoding_detail = _run_model_inference(
        model,
        video,
        use_nbest=use_nbest,
        nbest=nbest,
        beam_width=beam_width
    )
    language_hint = os.getenv("SWIN_VALLR_LANGUAGE_HINT")
    if language_hint:
        language_hint = language_hint.strip() or None
    refiner_mode = os.getenv("SWIN_VALLR_REFINER_MODE", "phoneme").strip().lower()
    if refiner_mode not in {"phoneme", "roman", "hybrid"}:
        refiner_mode = "phoneme"
    decoding = {
        "use_nbest": use_nbest,
        "nbest": nbest,
        "beam_width": beam_width,
        "language_hint": language_hint,
        "refiner_mode": refiner_mode
    }
    decoding["speaker_id"] = speaker_id
    decoding["multi_speaker"] = multi_speaker
    if decoding_detail:
        decoding.update(decoding_detail)
    _save_inference_artifacts(frames, rois, phonemes, text, speaker_id, hypotheses, decoding)
    LOGGER.info("Inference complete (phonemes=%d, text_len=%d).", len(phonemes), len(text))
    return phonemes, text, speaker_id, decoding


# =============================================================================
# Transcript Utilities
# =============================================================================

def _append_transcript_line(
    text: str,
    translation: Optional[str] = None,
    speaker_id: str = "Speaker 1",
    is_placeholder: bool = False
) -> None:
    if not text:
        return
    previous = st.session_state.transcript_lines[-1] if st.session_state.transcript_lines else None
    if previous and previous.get("text") == text and previous.get("translation") == translation:
        return
    st.session_state.transcript_lines.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "speaker": speaker_id,
        "text": text,
        "translation": translation or "",
        "placeholder": bool(is_placeholder)
    })


def _build_summary_input() -> str:
    lines = []
    for entry in st.session_state.transcript_lines:
        if entry.get("placeholder"):
            continue
        speaker = entry.get("speaker", "Speaker")
        text = entry.get("text", "")
        if text:
            lines.append(f"{speaker}: {text}")
    return " ".join(lines)


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the app header."""
    st.markdown(ANTI_GRAVITY_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3em; margin: 0;">🌌 SWIN-VALLR</h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.2em; margin-top: 10px;">
            Visual Speech Recognition • Powered by Swin Transformer + Qwen2
        </p>
    </div>
    """, unsafe_allow_html=True)


def _render_status_badge(label: str, is_online: bool, online_text: str, offline_text: str):
    """Render a single status badge."""
    status = "online" if is_online else "offline"
    text = online_text if is_online else offline_text

    st.markdown(f"""
    <div class="status-indicator status-{status}">
        <span class="pulse-dot {status}"></span>
        {label}: {text}
    </div>
    """, unsafe_allow_html=True)


def render_status_bar(model_loaded: bool, webcam_active: bool):
    """Render the status bar."""
    col1, col2, col3 = st.columns(3)

    with col1:
        _render_status_badge("Model", model_loaded, "Loaded", "Not Loaded")

    with col2:
        _render_status_badge("Webcam", webcam_active, "Active", "Inactive")

    with col3:
        st.markdown(f"""
        <div class="status-indicator status-online">
            <span class="pulse-dot online"></span>
            Backend: {get_backend_name()}
        </div>
        """, unsafe_allow_html=True)


def render_output_cards(phonemes: List[str], refined_text: str, translation_text: str = ""):
    """Render the output display cards."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="floating-card">
            <h3 style="margin-top: 0; font-size: 1em;">dY"S Raw Phonemes</h3>
        </div>
        """, unsafe_allow_html=True)

        phoneme_str = ' '.join(phonemes) if phonemes else "Waiting for speech..."
        st.markdown(f"""
        <div class="phoneme-box">{phoneme_str}</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="floating-card">
            <h3 style="margin-top: 0; font-size: 1em;">?o" Refined Text</h3>
        </div>
        """, unsafe_allow_html=True)

        text = refined_text if refined_text else "Waiting for speech..."
        st.markdown(f"""
        <div class="refined-text-box">{text}</div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="floating-card">
            <h3 style="margin-top: 0; font-size: 1em;">Translation</h3>
        </div>
        """, unsafe_allow_html=True)

        text = translation_text if translation_text else "Translation disabled..."
        st.markdown(f"""
        <div class="refined-text-box">{text}</div>
        """, unsafe_allow_html=True)


def render_caption_box(text: str, translation: str = "", speaker: str = ""):
    """Render a caption box styled like YouTube captions."""
    if not text:
        text = "Waiting for speech..."
    speaker_prefix = f"{speaker}: " if speaker else ""
    st.markdown(
        f"""
        <div class="caption-box">
            <div class="caption-text">{speaker_prefix}{text}</div>
            <div class="caption-translation">{translation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_transcript_panel():
    """Render transcript history and summary."""
    st.markdown("### Transcript")
    if not st.session_state.transcript_lines:
        st.info("No transcript yet.")
    else:
        for entry in st.session_state.transcript_lines[-20:]:
            timestamp = entry.get("timestamp", "")
            speaker = entry.get("speaker", "Speaker")
            text = entry.get("text", "")
            translation = entry.get("translation", "")
            st.markdown(f"**{timestamp} {speaker}:** {text}")
            if translation:
                st.markdown(f"*{translation}*")

    if st.session_state.summary_text:
        st.markdown("### Summary")
        st.markdown(st.session_state.summary_text)


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.markdown("""
        <h2 style="font-size: 1.2em;">⚙️ Settings</h2>
        """, unsafe_allow_html=True)

        st.markdown("### Model")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="checkpoints/best_model.pt",
            help="Path to model checkpoint"
        )

        weight_source = st.radio(
            "Weights",
            ["Checkpoint", "Random (no weights)"],
            index=0,
            help="Choose whether to load a checkpoint or use random weights."
        )
        use_checkpoint = (weight_source == "Checkpoint")
        load_refiner = st.checkbox(
            "Load Qwen2 Refiner",
            value=True,
            help="Enable linguistic refinement (requires more VRAM)"
        )
        auto_load_model = st.checkbox(
            "Auto-load model when needed",
            value=True,
            help="Automatically load the model on first inference."
        )

        st.markdown("### Vision")
        multi_speaker = st.checkbox(
            "Enable multi-speaker selection",
            value=_env_bool("SWIN_VALLR_MULTI_SPEAKER", False),
            help="Pick the most active speaker based on mouth motion."
        )
        max_faces = st.slider(
            "Max faces",
            min_value=1,
            max_value=4,
            value=_env_int("SWIN_VALLR_MAX_FACES", 1, min_value=1)
        )
        align_mouth = st.checkbox(
            "Align mouth ROI",
            value=_env_bool("SWIN_VALLR_ALIGN_MOUTH", True),
            help="Rotate the mouth crop to normalize tilt."
        )
        profile_aware = st.checkbox(
            "Profile-aware ROI",
            value=_env_bool("SWIN_VALLR_PROFILE_AWARE", True),
            help="Expand ROI for side-profile angles."
        )
        with st.expander("Profile tuning"):
            profile_expand = st.slider(
                "Profile expand",
                min_value=1.0,
                max_value=2.0,
                value=_env_float("SWIN_VALLR_PROFILE_EXPAND", 1.4, min_value=1.0),
                step=0.05
            )
            profile_yaw_scale = st.slider(
                "Profile yaw scale",
                min_value=0.0,
                max_value=4.0,
                value=_env_float("SWIN_VALLR_PROFILE_YAW_SCALE", 2.5, min_value=0.0),
                step=0.1
            )

        if multi_speaker and max_faces < 2:
            st.info("Increase Max faces to enable multi-speaker selection.")
        if not MEDIAPIPE_AVAILABLE:
            st.info("MediaPipe is unavailable; using center-crop fallback.")
        st.caption("Vision settings apply when the model is loaded.")

        if st.button("Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                try:
                    selected_path = checkpoint_path if use_checkpoint and Path(checkpoint_path).exists() else None
                    allow_random_output = not use_checkpoint
                    st.session_state.model = load_model(
                        selected_path,
                        load_refiner=load_refiner,
                        use_checkpoint=use_checkpoint,
                        allow_random_output=allow_random_output
                    )
                    st.session_state.lip_processor = load_lip_processor(
                        max_faces=max_faces,
                        align_mouth=align_mouth,
                        profile_aware=profile_aware,
                        profile_expand=profile_expand,
                        profile_yaw_scale=profile_yaw_scale
                    )
                    st.success("Model loaded!")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")

        if use_checkpoint and checkpoint_path and not Path(checkpoint_path).exists():
            st.warning("Checkpoint not found; using random weights.")
        if not use_checkpoint:
            env_value = os.getenv("SWIN_VALLR_ALLOW_RANDOM_OUTPUT")
            if env_value is not None and env_value.strip() and not _allow_random_output():
                st.info("Random weights loaded; low-confidence outputs are suppressed by SWIN_VALLR_ALLOW_RANDOM_OUTPUT.")
            else:
                st.info("Random weights loaded; output will be noisy without training.")

        st.markdown("---")

        st.markdown("### Display")
        live_captions = st.checkbox(
            "Live captions (auto-run)",
            value=True,
            help="Automatically run inference while the camera is active."
        )
        inference_interval = st.slider(
            "Inference Interval",
            min_value=1,
            max_value=10,
            value=DEFAULT_INFERENCE_INTERVAL,
            help="Process every Nth frame"
        )

        st.markdown("---")

        st.markdown("### Decoding")
        use_nbest = st.checkbox(
            "Use N-best decoding",
            value=False,
            help="Generate multiple CTC hypotheses and let the refiner choose"
        )
        nbest = st.slider(
            "N-best candidates",
            min_value=1,
            max_value=5,
            value=3
        )
        beam_width = st.slider(
            "Beam width",
            min_value=3,
            max_value=20,
            value=10
        )

        st.markdown("---")

        st.markdown("### Translation")
        translation_enabled = st.checkbox(
            "Enable live translation",
            value=False,
            help="Translate refined text on the fly using a lightweight model."
        )
        options = get_translation_options()
        option_labels = [f"{opt.name} ({opt.code})" for opt in options]
        default_index = 0
        translation_target = options[default_index].code if options else ""
        if translation_enabled and not options:
            st.info("Translation options are unavailable.")
            translation_enabled = False
        if options:
            selection = st.selectbox(
                "Target language",
                options=option_labels,
                index=default_index
            )
            selection_idx = option_labels.index(selection)
            translation_target = options[selection_idx].code

        st.markdown("---")

        st.markdown("### Summary")
        if st.button("Summarize Transcript", use_container_width=True):
            with st.spinner("Summarizing..."):
                if not st.session_state.transcript_lines:
                    st.info("Transcript is empty.")
                else:
                    summary_input = _build_summary_input()
                    st.session_state.summary_text = summarize_text(summary_input)
        if st.button("Clear Transcript", use_container_width=True):
            st.session_state.transcript_lines = []
            st.session_state.summary_text = ""

        st.markdown("### 📖 About")
        st.markdown("""
        **Swin-VALLR** combines:
        - 🔍 Swin Transformer encoder
        - 🧠 Qwen2-0.5B language model
        - 📹 MediaPipe face detection

        Supports AMD (DirectML/ROCm) and NVIDIA (CUDA).
        """)

        return (
            checkpoint_path,
            use_checkpoint,
            load_refiner,
            auto_load_model,
            max_faces,
            align_mouth,
            profile_aware,
            profile_expand,
            profile_yaw_scale,
            live_captions,
            inference_interval,
            use_nbest,
            nbest,
            beam_width,
            translation_enabled,
            translation_target,
            multi_speaker
        )


# =============================================================================
# Main Application
# =============================================================================

def _create_webrtc_context(inference_interval: int):
    """Create and render the WebRTC video stream."""
    st.markdown("""
    <div class="video-container">
    """, unsafe_allow_html=True)

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    processor = VideoProcessor()
    processor.inference_interval = inference_interval

    ctx = webrtc_streamer(
        key="lip-reading",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.webrtc_active = bool(getattr(ctx.state, "playing", False)) if ctx else False
    return ctx


def _sync_video_processor(
    processor: VideoProcessor,
    model: Optional[SwinVALLR],
    lip_processor: Optional['MediaPipeProcessor'],
    device: torch.device,
    use_nbest: bool,
    nbest: int,
    beam_width: int,
    multi_speaker: bool,
    live_captions: bool
) -> None:
    processor.model = model
    processor.lip_processor = lip_processor
    processor.device = device
    processor.use_nbest = use_nbest
    processor.nbest = nbest
    processor.beam_width = beam_width
    processor.multi_speaker = multi_speaker
    processor.live_inference = live_captions


def _drain_live_results(
    ctx,
    translation_enabled: bool,
    translation_target: str
) -> None:
    if not ctx or not ctx.video_processor:
        return
    while True:
        try:
            payload = ctx.video_processor.result_queue.get_nowait()
        except queue.Empty:
            break
        text = payload.get("text", "")
        phonemes = payload.get("phonemes", [])
        speaker_id = payload.get("speaker", "Speaker 1")
        placeholder = bool(payload.get("placeholder"))
        translation = ""
        if translation_enabled and text and not placeholder:
            translation = translate_text(text, translation_target)
        st.session_state.phoneme_buffer = phonemes
        st.session_state.text_buffer = text
        st.session_state.translation_buffer = translation
        st.session_state.speaker_buffer = speaker_id
        _append_transcript_line(text, translation, speaker_id, is_placeholder=placeholder)


def _render_webrtc_controls(
    ctx,
    use_nbest: bool,
    nbest: int,
    beam_width: int,
    translation_enabled: bool,
    translation_target: str,
    multi_speaker: bool,
    checkpoint_path: str,
    use_checkpoint: bool,
    load_refiner: bool,
    auto_load_model: bool,
    live_captions: bool,
    max_faces: int,
    align_mouth: bool,
    profile_aware: bool,
    profile_expand: float,
    profile_yaw_scale: float
):
    """Render inference controls and outputs for WebRTC mode."""
    if live_captions:
        st.info("Live captions are running while the camera is active.")
    if st.button("Run Inference", use_container_width=True):
        if st.session_state.model is None and auto_load_model:
            with st.spinner("Loading model..."):
                try:
                    selected_path = checkpoint_path if use_checkpoint and Path(checkpoint_path).exists() else None
                    allow_random_output = not use_checkpoint
                    st.session_state.model = load_model(
                        selected_path,
                        load_refiner=load_refiner,
                        use_checkpoint=use_checkpoint,
                        allow_random_output=allow_random_output
                    )
                    st.session_state.lip_processor = load_lip_processor(
                        max_faces=max_faces,
                        align_mouth=align_mouth,
                        profile_aware=profile_aware,
                        profile_expand=profile_expand,
                        profile_yaw_scale=profile_yaw_scale
                    )
                except Exception as exc:
                    log_exception(LOGGER, "Auto-load model failed.")
                    st.error(f"Failed to load model: {exc}")
                    return

        if st.session_state.model is not None and ctx.video_processor:
            frames = ctx.video_processor.frame_buffer.copy()
            if frames:
                LOGGER.info("Running inference on %d buffered frames.", len(frames))
                with st.spinner("Processing..."):
                    try:
                        phonemes, text, speaker_id, decoding = process_frames_for_inference(
                            frames,
                            st.session_state.lip_processor,
                            st.session_state.model,
                            get_device(),
                            use_nbest=use_nbest,
                            nbest=nbest,
                            beam_width=beam_width,
                            multi_speaker=multi_speaker
                        )
                    except Exception as exc:
                        log_exception(LOGGER, "Inference failed in WebRTC mode.")
                        st.error(f"Inference failed: {exc}")
                        return

                    placeholder = bool(decoding.get("untrained_guard")) if decoding else False
                    if placeholder and not text:
                        text = (
                            "Untrained model: output suppressed. "
                            "Load a checkpoint or set SWIN_VALLR_ALLOW_RANDOM_OUTPUT=true."
                        )
                    translation = ""
                    if translation_enabled and text and not placeholder:
                        translation = translate_text(text, translation_target)
                    st.session_state.phoneme_buffer = phonemes
                    st.session_state.text_buffer = text
                    st.session_state.translation_buffer = translation
                    st.session_state.speaker_buffer = speaker_id
                    _append_transcript_line(text, translation, speaker_id, is_placeholder=placeholder)
            else:
                LOGGER.warning("Inference requested with empty frame buffer.")
                st.warning("No frames captured yet")
        else:
            LOGGER.warning("Inference requested but model or webcam not ready.")
            st.warning("Please load the model first")

    st.markdown("### Results")
    render_output_cards(
        st.session_state.phoneme_buffer,
        st.session_state.text_buffer,
        st.session_state.translation_buffer
    )


def _render_webrtc_mode(
    inference_interval: int,
    live_captions: bool,
    use_nbest: bool,
    nbest: int,
    beam_width: int,
    translation_enabled: bool,
    translation_target: str,
    multi_speaker: bool,
    checkpoint_path: str,
    use_checkpoint: bool,
    load_refiner: bool,
    auto_load_model: bool,
    max_faces: int,
    align_mouth: bool,
    profile_aware: bool,
    profile_expand: float,
    profile_yaw_scale: float
):
    """Render the WebRTC capture and inference layout."""
    col1, col2 = st.columns([2, 1])

    with col1:
        ctx = _create_webrtc_context(inference_interval)
        render_caption_box(
            st.session_state.text_buffer,
            st.session_state.translation_buffer,
            st.session_state.speaker_buffer
        )

    with col2:
        _render_webrtc_controls(
            ctx,
            use_nbest,
            nbest,
            beam_width,
            translation_enabled,
            translation_target,
            multi_speaker,
            checkpoint_path,
            use_checkpoint,
            load_refiner,
            auto_load_model,
            live_captions,
            max_faces,
            align_mouth,
            profile_aware,
            profile_expand,
            profile_yaw_scale
        )

    if ctx and getattr(ctx.state, "playing", False):
        if st.session_state.model is None and auto_load_model:
            with st.spinner("Loading model..."):
                try:
                    selected_path = checkpoint_path if use_checkpoint and Path(checkpoint_path).exists() else None
                    allow_random_output = not use_checkpoint
                    st.session_state.model = load_model(
                        selected_path,
                        load_refiner=load_refiner,
                        use_checkpoint=use_checkpoint,
                        allow_random_output=allow_random_output
                    )
                    st.session_state.lip_processor = load_lip_processor(
                        max_faces=max_faces,
                        align_mouth=align_mouth,
                        profile_aware=profile_aware,
                        profile_expand=profile_expand,
                        profile_yaw_scale=profile_yaw_scale
                    )
                except Exception as exc:
                    log_exception(LOGGER, "Auto-load model failed (live).")
                    st.error(f"Failed to load model: {exc}")
        if ctx.video_processor:
            _sync_video_processor(
                ctx.video_processor,
                st.session_state.model,
                st.session_state.lip_processor,
                get_device(),
                use_nbest,
                nbest,
                beam_width,
                multi_speaker,
                live_captions
            )
            _drain_live_results(ctx, translation_enabled, translation_target)


def _render_file_upload_mode(
    use_nbest: bool,
    nbest: int,
    beam_width: int,
    translation_enabled: bool,
    translation_target: str,
    multi_speaker: bool,
    checkpoint_path: str,
    use_checkpoint: bool,
    load_refiner: bool,
    auto_load_model: bool,
    max_faces: int,
    align_mouth: bool,
    profile_aware: bool,
    profile_expand: float,
    profile_yaw_scale: float
):
    """Render the file upload fallback flow."""
    st.warning("WebRTC not available. Using file upload mode.")
    LOGGER.info("Using file upload mode (WebRTC unavailable).")

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'webm']
    )

    if uploaded_file is not None:
        temp_path = Path("temp_video.mp4")
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            st.video(str(temp_path))

            if st.button("Run Inference", use_container_width=True):
                if st.session_state.model is None and auto_load_model:
                    with st.spinner("Loading model..."):
                        try:
                            selected_path = checkpoint_path if use_checkpoint and Path(checkpoint_path).exists() else None
                            allow_random_output = not use_checkpoint
                            st.session_state.model = load_model(
                                selected_path,
                                load_refiner=load_refiner,
                                use_checkpoint=use_checkpoint,
                                allow_random_output=allow_random_output
                            )
                            st.session_state.lip_processor = load_lip_processor(
                                max_faces=max_faces,
                                align_mouth=align_mouth,
                                profile_aware=profile_aware,
                                profile_expand=profile_expand,
                                profile_yaw_scale=profile_yaw_scale
                            )
                        except Exception as exc:
                            log_exception(LOGGER, "Auto-load model failed (upload).")
                            st.error(f"Failed to load model: {exc}")
                            return

                if st.session_state.model is not None:
                    with st.spinner("Processing video..."):
                        LOGGER.info("Processing uploaded video for inference.")
                        cap = cv2.VideoCapture(str(temp_path))
                        frames = []
                        progress = tqdm(total=TARGET_FRAMES, desc="Reading video frames", unit="frame", leave=False)
                        try:
                            if not cap.isOpened():
                                st.error("Could not open the uploaded video.")
                                LOGGER.warning("Uploaded video could not be opened.")
                                return
                            while len(frames) < TARGET_FRAMES:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frames.append(frame)
                                progress.update(1)
                        finally:
                            progress.close()
                            cap.release()

                        if frames:
                            LOGGER.info("Read %d frames from uploaded video.", len(frames))
                            try:
                                phonemes, text, speaker_id, decoding = process_frames_for_inference(
                                    frames,
                                    st.session_state.lip_processor,
                                    st.session_state.model,
                                    get_device(),
                                    use_nbest=use_nbest,
                                    nbest=nbest,
                                    beam_width=beam_width,
                                    multi_speaker=multi_speaker
                                )
                            except Exception as exc:
                                log_exception(LOGGER, "Inference failed for uploaded video.")
                                st.error(f"Inference failed: {exc}")
                                return
                            placeholder = bool(decoding.get("untrained_guard")) if decoding else False
                            if placeholder and not text:
                                text = (
                                    "Untrained model: output suppressed. "
                                    "Load a checkpoint or set SWIN_VALLR_ALLOW_RANDOM_OUTPUT=true."
                                )
                            translation = ""
                            if translation_enabled and text and not placeholder:
                                translation = translate_text(text, translation_target)
                            st.session_state.translation_buffer = translation
                            st.session_state.speaker_buffer = speaker_id
                            _append_transcript_line(text, translation, speaker_id, is_placeholder=placeholder)

                            render_output_cards(phonemes, text, translation)
                            render_caption_box(text, translation, speaker_id)
                        else:
                            LOGGER.warning("No frames read from uploaded video.")
                            st.error("Could not read video frames")
                else:
                    LOGGER.warning("Upload inference requested without a loaded model.")
                    st.warning("Please load the model first")
        finally:
            if temp_path.exists():
                temp_path.unlink()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Swin-VALLR • Visual Speech Recognition",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    log_system_info(LOGGER)
    LOGGER.info("App main entrypoint invoked.")
    init_session_state()
    render_header()
    if not MEDIAPIPE_AVAILABLE:
        st.warning("MediaPipe not available; using center-crop fallback.")

    (
        checkpoint_path,
        use_checkpoint,
        load_refiner,
        auto_load_model,
        max_faces,
        align_mouth,
        profile_aware,
        profile_expand,
        profile_yaw_scale,
        live_captions,
        inference_interval,
        use_nbest,
        nbest,
        beam_width,
        translation_enabled,
        translation_target,
        multi_speaker
    ) = render_sidebar()

    model_loaded = st.session_state.model is not None
    render_status_bar(model_loaded, st.session_state.webrtc_active)

    st.markdown("---")

    if WEBRTC_AVAILABLE:
        _render_webrtc_mode(
            inference_interval,
            live_captions,
            use_nbest,
            nbest,
            beam_width,
            translation_enabled,
            translation_target,
            multi_speaker,
            checkpoint_path,
            use_checkpoint,
            load_refiner,
            auto_load_model,
            max_faces,
            align_mouth,
            profile_aware,
            profile_expand,
            profile_yaw_scale
        )
    else:
        _render_file_upload_mode(
            use_nbest,
            nbest,
            beam_width,
            translation_enabled,
            translation_target,
            multi_speaker,
            checkpoint_path,
            use_checkpoint,
            load_refiner,
            auto_load_model,
            max_faces,
            align_mouth,
            profile_aware,
            profile_expand,
            profile_yaw_scale
        )

    
    render_transcript_panel()
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.5); padding: 20px;">
        <p>Swin-VALLR • Visual Speech Recognition System</p>
        <p style="font-size: 0.8em;">Powered by Swin Transformer + Qwen2-0.5B</p>
    </div>
    """, unsafe_allow_html=True)


def _has_streamlit_context() -> bool:
    """Return True when running under `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return True
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    try:
        if not _has_streamlit_context():
            print("This app must be run with: streamlit run app.py")
            sys.exit(0)
        main()
    except Exception:
        log_exception(LOGGER, "App crashed with an unhandled exception.")
        raise
