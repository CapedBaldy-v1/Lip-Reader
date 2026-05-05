"""
Swin-VALLR Data Pipeline
=========================
Dataset handling, preprocessing, and data loading for Visual Speech Recognition.

Features:
- Interactive dataset download (LRS2/LRS3)
- MediaPipe Face Mesh for lip ROI extraction
- Kalman filter for mouth stabilization
- Curriculum learning sampler
- Visual-only prefilter for custom dataset quality control (no audio)
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, asdict
from enum import Enum, auto
from collections import Counter

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from logging_utils import setup_logging, log_exception, log_system_info
from artifact_utils import (
    get_artifact_root,
    get_useful_dir,
    save_json,
    save_csv,
    save_line_plot,
    save_montage,
    save_text
)


LOGGER = setup_logging("data_pipeline")

try:
    import mediapipe as mp
    # Try old API first (mp.solutions)
    if hasattr(mp, "solutions"):
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_NEW_API = False
    else:
        # Try new API (mp.tasks)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            # Create compatibility wrapper
            class _SolutionsCompat:
                class face_mesh:
                    @staticmethod
                    def FaceMesh(*args, **kwargs):
                        # Wrapper will be created in MediaPipeProcessor
                        return None
            mp.solutions = _SolutionsCompat()
            mp._tasks_python = mp_python
            mp._tasks_vision = mp_vision
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_NEW_API = True
            print("[DataPipeline] Using new MediaPipe tasks API (0.10+)")
        except Exception as exc:
            MEDIAPIPE_AVAILABLE = False
            MEDIAPIPE_NEW_API = False
            print("[DataPipeline] MediaPipe not available - lip extraction disabled")
            LOGGER.warning("MediaPipe not available: %s", exc)
except Exception as exc:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_NEW_API = False
    print("[DataPipeline] MediaPipe not available - lip extraction disabled")
    LOGGER.warning("MediaPipe not available - lip extraction disabled. %s", exc)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ROI_SIZE = 96
DEFAULT_TARGET_FPS = 25
DEFAULT_CLIP_DURATION = 2.0
DEFAULT_NUM_FRAMES = 50

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # ROI settings
    roi_size: int = DEFAULT_ROI_SIZE
    target_fps: int = DEFAULT_TARGET_FPS
    clip_duration: float = DEFAULT_CLIP_DURATION
    num_frames: int = DEFAULT_NUM_FRAMES

    # Normalization
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # Data augmentation
    horizontal_flip: bool = True
    random_crop: bool = True
    random_crop_prob: float = 0.3
    random_crop_shift: float = 0.08
    random_crop_scale: float = 0.1
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    rotation_prob: float = 0.15
    max_rotation_deg: float = 5.0
    profile_warp_prob: float = 0.08
    profile_warp_strength: float = 0.12
    temporal_mask_prob: float = 0.2
    temporal_mask_max_len: int = 8
    speed_perturb_prob: float = 0.2
    speed_perturb_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)

    # Robustness augmentations (paper-1+)
    random_erasing_prob: float = 0.2
    random_erasing_scale: Tuple[float, float] = (0.02, 0.12)
    random_erasing_ratio: Tuple[float, float] = (0.3, 3.3)
    random_erasing_value: float = 0.0
    gaussian_blur_prob: float = 0.1
    gaussian_blur_kernel: Tuple[int, int] = (3, 5)
    gaussian_noise_prob: float = 0.15
    gaussian_noise_std: float = 0.05
    downscale_prob: float = 0.1
    downscale_range: Tuple[float, float] = (0.7, 1.0)
    frame_dropout_prob: float = 0.1
    frame_dropout_max_ratio: float = 0.15

    # Beard + dark occlusion handling
    beard_occlusion_prob: float = 0.15
    beard_occlusion_strength: float = 0.6
    beard_occlusion_height_ratio: float = 0.35
    beard_occlusion_noise: float = 0.05
    black_bar_prob: float = 0.08
    black_bar_height_ratio: float = 0.2
    black_bar_value: float = -3.0

    # Augmentation schedule
    augmentation_ramp_epochs: int = 5
    augmentation_max_strength: float = 1.0

    # Curriculum learning phases
    curriculum_phases: Tuple[int, ...] = (5, 20)

    # Multilingual balancing (paper-11+)
    language_balance: bool = False
    language_balance_power: float = 0.7

    # Training quality filters. Zero disables each filter.
    max_words: int = 0
    max_phoneme_length: int = 0
    max_ctc_required_frames: int = 0
    min_word_confidence: float = 0.0
    min_face_frame_rate: float = 0.0
    min_mouth_motion: float = 0.0


@dataclass
class VisualFilterConfig:
    """Visual-only prefilter configuration for custom dataset builds."""
    enabled: bool = False
    sample_frames: int = 100
    sample_stride: int = 1
    min_face_rate: float = 0.7
    min_lip_size_ratio: float = 0.03
    min_openness_std: float = 0.004
    min_motion: float = 0.003
    max_motion: float = 0.08
    min_sharpness: float = 10.0
    preview_limit: int = 6
    preview_frames: int = 16
    preview_size: int = 96


@dataclass
class LipMetrics:
    """Per-frame lip measurements for visual filtering."""
    center: Tuple[float, float]
    size: float
    openness: float
    bbox: Tuple[int, int, int, int]


@dataclass
class VisualSpeechMetrics:
    """Aggregated visual speech quality metrics for a video."""
    frame_samples: int
    detections: int
    face_rate: float
    avg_lip_size: float
    lip_size_ratio: float
    openness_mean: float
    openness_std: float
    motion_mean: float
    motion_std: float
    sharpness_mean: float
    sharpness_std: float
    score: float


class CurriculumPhase(Enum):
    """Curriculum learning phases."""
    SINGLE_WORDS = auto()
    SHORT_SENTENCES = auto()
    FULL_DATASET = auto()


# =============================================================================
# MediaPipe Lip Extractor
# =============================================================================

class MediaPipeProcessor:
    """
    MediaPipe-based lip ROI extraction with Kalman filter stabilization.
    """

    # Lip landmark indices in MediaPipe Face Mesh
    # Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    # Inner lip: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
    LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    LIP_LEFT = 61
    LIP_RIGHT = 291

    def __init__(
        self,
        roi_size: int = DEFAULT_ROI_SIZE,
        stabilize: bool = True,
        crop_mode: str = "adaptive",
        crop_scale: float = 2.0,
        smooth_sigma: float = 4.0,
        adaptive_motion_ratio: float = 0.1,
        max_faces: int = 1,
        align_mouth: bool = True,
        profile_aware: bool = True,
        profile_expand: float = 1.4,
        profile_yaw_scale: float = 2.5
    ):
        self.roi_size = roi_size
        self.stabilize = stabilize
        self.crop_mode = crop_mode
        self.crop_scale = crop_scale
        self.smooth_sigma = smooth_sigma
        self.adaptive_motion_ratio = adaptive_motion_ratio
        self.max_faces = max(1, int(max_faces))
        self.align_mouth = align_mouth
        self.profile_aware = profile_aware
        self.profile_expand = max(1.0, float(profile_expand))
        self.profile_yaw_scale = float(profile_yaw_scale)

        if not MEDIAPIPE_AVAILABLE:
            LOGGER.error("MediaPipeProcessor init requested but MediaPipe is missing.")
            raise RuntimeError("MediaPipe is not installed")

        valid_modes = {"frame", "fixed", "smooth", "adaptive"}
        if self.crop_mode not in valid_modes:
            LOGGER.warning("Invalid crop_mode=%s, falling back to 'fixed'.", self.crop_mode)
            self.crop_mode = "fixed"

        self._create_face_mesh()

        if stabilize:
            self._init_kalman_filter()
        LOGGER.info(
            "MediaPipeProcessor initialized (roi_size=%s, stabilize=%s, crop_mode=%s).",
            roi_size,
            stabilize,
            self.crop_mode
        )

    def _create_face_mesh(self) -> None:
        """Initialize the MediaPipe Face Mesh model."""
        global MEDIAPIPE_NEW_API
        
        if hasattr(mp, '_tasks_vision'):
            # New API (MediaPipe 0.10+)
            import tempfile
            import urllib.request
            
            # Download face landmarker model
            model_path = Path(tempfile.gettempdir()) / 'face_landmarker.task'
            if not model_path.exists():
                print("[MediaPipe] Downloading face landmarker model...")
                model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
                urllib.request.urlretrieve(model_url, model_path)
                print(f"[MediaPipe] Model downloaded to {model_path}")
            
            # Create FaceLandmarker
            BaseOptions = mp._tasks_python.BaseOptions
            FaceLandmarker = mp._tasks_vision.FaceLandmarker
            FaceLandmarkerOptions = mp._tasks_vision.FaceLandmarkerOptions
            VisionRunningMode = mp._tasks_vision.RunningMode
            
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=self.max_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            
            self.face_mesh = FaceLandmarker.create_from_options(options)
            self.mp_face_mesh = None
            self._new_api = True
            self._frame_timestamp_ms = 0
        else:
            # Old API (MediaPipe < 0.10)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._new_api = False
    
    def _process_frame(self, rgb_frame: np.ndarray):
        """Process frame with MediaPipe (handles both old and new API)."""
        if self._new_api:
            # New API requires mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self._frame_timestamp_ms += 33  # ~30fps
            result = self.face_mesh.detect_for_video(mp_image, self._frame_timestamp_ms)
            
            # Convert to old API format
            class FakeLandmarks:
                def __init__(self, landmarks):
                    self.landmark = landmarks
            
            class FakeResults:
                def __init__(self):
                    self.multi_face_landmarks = []
            
            fake_results = FakeResults()
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    fake_results.multi_face_landmarks.append(FakeLandmarks(face_landmarks))
            
            return fake_results
        else:
            # Old API
            return self._process_frame(rgb_frame)

    def _init_kalman_filter(self):
        """Initialize Kalman filter for mouth center tracking."""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman_initialized = False

    def _get_lip_center(self, landmarks, img_w: int, img_h: int) -> Tuple[int, int]:
        """Get center of lip landmarks."""
        all_lip = self.LIP_OUTER + self.LIP_INNER

        x_coords = [landmarks[i].x * img_w for i in all_lip]
        y_coords = [landmarks[i].y * img_h for i in all_lip]

        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        return center_x, center_y

    def _get_lip_bbox(self, landmarks, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Get bounding box around lips with margin."""
        all_lip = self.LIP_OUTER + self.LIP_INNER

        x_coords = [landmarks[i].x * img_w for i in all_lip]
        y_coords = [landmarks[i].y * img_h for i in all_lip]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        width = max_x - min_x
        height = max_y - min_y
        margin = max(width, height) * 0.5

        return (
            int(min_x - margin),
            int(min_y - margin),
            int(max_x + margin),
            int(max_y + margin)
        )

    def _get_lip_size(self, landmarks, img_w: int, img_h: int) -> float:
        """Get the lip bounding size for crop scaling."""
        all_lip = self.LIP_OUTER + self.LIP_INNER

        x_coords = [landmarks[i].x * img_w for i in all_lip]
        y_coords = [landmarks[i].y * img_h for i in all_lip]

        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return float(max(width, height))

    def _metrics_from_landmarks(
        self,
        landmarks,
        img_w: int,
        img_h: int
    ) -> LipMetrics:
        center_x, center_y = self._get_lip_center(landmarks, img_w, img_h)
        size = self._get_lip_size(landmarks, img_w, img_h)
        bbox = self._get_lip_bbox(landmarks, img_w, img_h)
        upper = landmarks[13]
        lower = landmarks[14]
        openness = abs((lower.y - upper.y) * img_h)
        return LipMetrics(
            center=(float(center_x), float(center_y)),
            size=float(size),
            openness=float(openness),
            bbox=bbox
        )

    def _estimate_yaw(self, landmarks) -> float:
        left = landmarks[self.LIP_LEFT]
        right = landmarks[self.LIP_RIGHT]
        return float(right.z - left.z)

    def _stabilize_center(self, center_x: int, center_y: int) -> Tuple[int, int]:
        """Apply Kalman filter to stabilize center position."""
        if not self.stabilize:
            return center_x, center_y

        measurement = np.array([[center_x], [center_y]], dtype=np.float32)

        if not self.kalman_initialized:
            self.kalman.statePost = np.array([
                [center_x], [center_y], [0], [0]
            ], dtype=np.float32)
            self.kalman_initialized = True
            return center_x, center_y

        self.kalman.predict()
        corrected = self.kalman.correct(measurement)

        return int(corrected[0, 0]), int(corrected[1, 0])

    def _extract_frame_stats(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int], float, float]]:
        """Extract lip center, size, and mouth angle for a frame."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._process_frame(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        center_x, center_y = self._get_lip_center(landmarks, width, height)
        center_x, center_y = self._stabilize_center(center_x, center_y)
        size = self._get_lip_size(landmarks, width, height)
        if self.profile_aware:
            yaw = abs(self._estimate_yaw(landmarks))
            expand = min(self.profile_expand, 1.0 + yaw * self.profile_yaw_scale)
            size *= expand
        left = landmarks[self.LIP_LEFT]
        right = landmarks[self.LIP_RIGHT]
        dx = (right.x - left.x) * width
        dy = (right.y - left.y) * height
        angle = math.degrees(math.atan2(dy, dx))
        return (center_x, center_y), size, angle

    def extract_lip_metrics(self, frame: np.ndarray) -> Optional[LipMetrics]:
        """Extract lip metrics for visual-only filtering."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._process_frame(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        return self._metrics_from_landmarks(landmarks, width, height)

    def extract_lip_metrics_multi(self, frame: np.ndarray) -> List[LipMetrics]:
        """Extract lip metrics for all detected faces."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._process_frame(rgb_frame)

        if not results.multi_face_landmarks:
            return []

        metrics_list: List[LipMetrics] = []
        for landmarks in results.multi_face_landmarks:
            metrics_list.append(self._metrics_from_landmarks(landmarks.landmark, width, height))
        return metrics_list

    def extract_face_observations(self, frame: np.ndarray) -> List[Tuple[LipMetrics, object]]:
        """Extract lip metrics and landmarks for all detected faces."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._process_frame(rgb_frame)

        if not results.multi_face_landmarks:
            return []

        observations: List[Tuple[LipMetrics, object]] = []
        for landmarks in results.multi_face_landmarks:
            metrics = self._metrics_from_landmarks(landmarks.landmark, width, height)
            observations.append((metrics, landmarks.landmark))
        return observations

    def _fill_missing_centers(
        self,
        centers: List[Optional[Tuple[int, int]]],
        fallback: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Fill missing centers with a fallback value."""
        filled = []
        last = fallback
        for center in centers:
            if center is None:
                filled.append(last)
            else:
                last = center
                filled.append(center)
        return [(float(x), float(y)) for x, y in filled]

    def _fill_missing_sizes(
        self,
        sizes: List[Optional[float]],
        fallback: float
    ) -> List[float]:
        """Fill missing sizes with a fallback value."""
        filled = []
        last = fallback
        for size in sizes:
            if size is None:
                filled.append(last)
            else:
                last = size
                filled.append(size)
        return [float(s) for s in filled]

    def _fill_missing_angles(
        self,
        angles: List[Optional[float]],
        fallback: float
    ) -> List[float]:
        """Fill missing angles with a fallback value."""
        filled = []
        last = fallback
        for angle in angles:
            if angle is None:
                filled.append(last)
            else:
                last = angle
                filled.append(angle)
        return [float(a) for a in filled]

    def _gaussian_smooth_1d(self, values: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing to a 1D sequence."""
        if sigma <= 0 or len(values) < 3:
            return values

        radius = int(max(1, round(sigma * 3)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()

        padded = np.pad(values, (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _compute_fixed_crop(
        self,
        centers: np.ndarray,
        sizes: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute a robust fixed crop center and size."""
        center_median = np.median(centers, axis=0)
        distances = np.linalg.norm(centers - center_median, axis=1)
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))

        threshold = max(5.0, median_dist + 3.0 * mad)
        inliers = distances <= threshold

        if np.any(inliers):
            centers = centers[inliers]
            sizes = sizes[inliers]

        fixed_center = centers.mean(axis=0)
        fixed_size = float(np.percentile(sizes, 95))
        return fixed_center, max(1.0, fixed_size)

    def _compute_fixed_angle(self, angles: np.ndarray) -> float:
        """Compute a robust fixed mouth angle."""
        if angles.size == 0:
            return 0.0
        return float(np.median(angles))

    def _select_crop_mode(
        self,
        centers: np.ndarray,
        fixed_center: np.ndarray
    ) -> str:
        """Pick the crop mode based on motion in the clip."""
        if self.crop_mode != "adaptive":
            return self.crop_mode

        if centers.size == 0:
            return "fixed"

        avg_motion = float(np.mean(np.linalg.norm(centers - fixed_center, axis=1)))
        threshold = float(self.roi_size) * self.adaptive_motion_ratio
        return "smooth" if avg_motion > threshold else "fixed"

    def _compute_crop_size(self, lip_size: float, frame_shape: Tuple[int, int]) -> int:
        """Compute the crop size in pixels."""
        height, width = frame_shape
        target = max(self.roi_size, int(round(lip_size * self.crop_scale)))
        return max(1, min(target, height, width))

    def _crop_with_center(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        lip_size: float
    ) -> Optional[np.ndarray]:
        """Crop the frame using a given center and lip size."""
        height, width = frame.shape[:2]
        crop_size = self._compute_crop_size(lip_size, (height, width))

        half = crop_size / 2.0
        center_x, center_y = center
        x1 = int(round(center_x - half))
        y1 = int(round(center_y - half))

        x1 = max(0, min(x1, width - crop_size))
        y1 = max(0, min(y1, height - crop_size))

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        roi = frame[y1:y2, x1:x2]

        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return None

        return cv2.resize(roi, (self.roi_size, self.roi_size))

    def extract_lip_roi_from_landmarks(self, frame: np.ndarray, landmarks) -> Optional[np.ndarray]:
        """Extract a lip ROI for a specific face landmarks set."""
        height, width = frame.shape[:2]
        center_x, center_y = self._get_lip_center(landmarks, width, height)
        lip_size = self._get_lip_size(landmarks, width, height)

        if self.profile_aware:
            yaw = abs(self._estimate_yaw(landmarks))
            expand = min(self.profile_expand, 1.0 + yaw * self.profile_yaw_scale)
            lip_size *= expand

        roi = self._crop_with_center(frame, (center_x, center_y), lip_size)
        if roi is None:
            return None

        if self.align_mouth:
            left = landmarks[self.LIP_LEFT]
            right = landmarks[self.LIP_RIGHT]
            dx = (right.x - left.x) * width
            dy = (right.y - left.y) * height
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) > 1.0:
                center = (roi.shape[1] // 2, roi.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))

        return roi

    def extract_lip_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract lip ROI from a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Lip ROI (roi_size, roi_size, 3) or None if face not detected
        """
        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._process_frame(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        center_x, center_y = self._get_lip_center(landmarks, width, height)
        center_x, center_y = self._stabilize_center(center_x, center_y)
        lip_size = self._get_lip_size(landmarks, width, height)

        if self.profile_aware:
            yaw = abs(self._estimate_yaw(landmarks))
            expand = min(self.profile_expand, 1.0 + yaw * self.profile_yaw_scale)
            lip_size *= expand

        roi = self._crop_with_center(frame, (center_x, center_y), lip_size)
        if roi is None:
            return None

        if self.align_mouth:
            left = landmarks[self.LIP_LEFT]
            right = landmarks[self.LIP_RIGHT]
            dx = (right.x - left.x) * width
            dy = (right.y - left.y) * height
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) > 1.0:
                center = (roi.shape[1] // 2, roi.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))

        return roi

    def process_video(
        self,
        video_path: str,
        target_fps: int = DEFAULT_TARGET_FPS,
        num_frames: int = DEFAULT_NUM_FRAMES
    ) -> Optional[np.ndarray]:
        """
        Process a video file and extract lip ROIs.

        Args:
            video_path: Path to video file
            target_fps: Target FPS for output
            num_frames: Number of frames to extract

        Returns:
            Lip ROI sequence (num_frames, roi_size, roi_size, 3) or None
        """
        LOGGER.info("Processing video: %s", video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            LOGGER.warning("Failed to open video: %s", video_path)
            print(f"[MediaPipeProcessor] Failed to open video: {video_path}")
            return None

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if orig_fps <= 0 or total_frames <= 0:
            cap.release()
            LOGGER.warning("Video FPS or frame count invalid for %s.", video_path)
            return None

        frame_interval = orig_fps / target_fps

        if self.stabilize:
            self._init_kalman_filter()

        frames: List[np.ndarray] = []
        centers: List[Optional[Tuple[int, int]]] = []
        sizes: List[Optional[float]] = []
        angles: List[Optional[float]] = []
        frame_idx = 0
        progress = tqdm(total=num_frames, desc="Extracting frames", unit="frame", leave=False)

        try:
            while len(frames) < num_frames:
                target_frame = int(frame_idx * frame_interval)

                if target_frame >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()

                if not ret:
                    break

                stats = self._extract_frame_stats(frame)

                if stats is None:
                    centers.append(None)
                    sizes.append(None)
                    angles.append(None)
                else:
                    center, size, angle = stats
                    centers.append(center)
                    sizes.append(size)
                    angles.append(angle)

                frames.append(frame)

                progress.update(1)
                frame_idx += 1

            while len(frames) < num_frames:
                if frames:
                    frames.append(frames[-1].copy())
                    centers.append(centers[-1])
                    sizes.append(sizes[-1])
                    angles.append(angles[-1])
                else:
                    frames.append(np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8))
                    centers.append(None)
                    sizes.append(None)
                    angles.append(None)
                progress.update(1)
        finally:
            cap.release()
            progress.close()

        if not frames:
            empty = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            rois = [empty.copy() for _ in range(num_frames)]
            return np.stack(rois, axis=0)

        valid_centers = [c for c in centers if c is not None]
        valid_sizes = [s for s in sizes if s is not None]
        valid_angles = [a for a in angles if a is not None]

        if not valid_centers or not valid_sizes:
            empty = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            rois = [empty.copy() for _ in range(num_frames)]
            return np.stack(rois, axis=0)

        valid_centers_arr = np.array(valid_centers, dtype=np.float32)
        valid_sizes_arr = np.array(valid_sizes, dtype=np.float32)
        fixed_center, fixed_size = self._compute_fixed_crop(valid_centers_arr, valid_sizes_arr)
        fixed_angle = self._compute_fixed_angle(np.array(valid_angles, dtype=np.float32))

        filled_centers = self._fill_missing_centers(centers, tuple(fixed_center))
        filled_sizes = self._fill_missing_sizes(sizes, fixed_size)
        filled_angles = self._fill_missing_angles(angles, fixed_angle)

        centers_arr = np.array(filled_centers, dtype=np.float32)
        sizes_arr = np.array(filled_sizes, dtype=np.float32)
        angles_arr = np.array(filled_angles, dtype=np.float32)

        crop_mode = self._select_crop_mode(centers_arr, fixed_center)
        LOGGER.debug("Crop mode selected: %s", crop_mode)

        if crop_mode == "smooth":
            smooth_x = self._gaussian_smooth_1d(centers_arr[:, 0], self.smooth_sigma)
            smooth_y = self._gaussian_smooth_1d(centers_arr[:, 1], self.smooth_sigma)
            smooth_sizes = self._gaussian_smooth_1d(sizes_arr, self.smooth_sigma)
            smooth_centers = np.stack([smooth_x, smooth_y], axis=1)
            smooth_angles = None
            if self.align_mouth:
                smooth_angles = self._gaussian_smooth_1d(angles_arr, self.smooth_sigma)
        else:
            smooth_centers = None
            smooth_sizes = None
            smooth_angles = None

        rois: List[np.ndarray] = []
        for idx, frame in enumerate(frames[:num_frames]):
            if crop_mode == "fixed":
                center = (float(fixed_center[0]), float(fixed_center[1]))
                size = fixed_size
                angle = float(fixed_angle)
            elif crop_mode == "smooth" and smooth_centers is not None and smooth_sizes is not None:
                center = (float(smooth_centers[idx, 0]), float(smooth_centers[idx, 1]))
                size = float(smooth_sizes[idx])
                angle = float(smooth_angles[idx]) if smooth_angles is not None else float(angles_arr[idx])
            else:
                center = (float(centers_arr[idx, 0]), float(centers_arr[idx, 1]))
                size = float(sizes_arr[idx])
                angle = float(angles_arr[idx])

            roi = self._crop_with_center(frame, center, size)

            if roi is not None:
                if self.align_mouth and abs(angle) > 1.0:
                    center_pt = (roi.shape[1] // 2, roi.shape[0] // 2)
                    matrix = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
                    roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))
                rois.append(roi)
            elif rois:
                rois.append(rois[-1].copy())
            else:
                rois.append(np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8))

        rois = rois[:num_frames]

        LOGGER.info("Completed ROI extraction for %s (%d frames).", video_path, len(rois))
        return np.stack(rois, axis=0)

    def close(self):
        """Release resources."""
        self.face_mesh.close()
        LOGGER.info("MediaPipeProcessor resources released.")


# =============================================================================
# Visual-only Prefilter (Paper 11 inspired)
# =============================================================================

def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid %s=%s; using default %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid %s=%s; using default %s", name, value, default)
        return default


def _visual_filter_config_from_env() -> VisualFilterConfig:
    return VisualFilterConfig(
        enabled=_env_flag("SWIN_VALLR_VISUAL_FILTER"),
        sample_frames=_env_int("SWIN_VALLR_VISUAL_FILTER_FRAMES", 100),
        sample_stride=_env_int("SWIN_VALLR_VISUAL_FILTER_STRIDE", 1),
        min_face_rate=_env_float("SWIN_VALLR_VISUAL_FILTER_MIN_FACE_RATE", 0.7),
        min_lip_size_ratio=_env_float("SWIN_VALLR_VISUAL_FILTER_MIN_LIP_RATIO", 0.03),
        min_openness_std=_env_float("SWIN_VALLR_VISUAL_FILTER_MIN_OPENNESS", 0.004),
        min_motion=_env_float("SWIN_VALLR_VISUAL_FILTER_MIN_MOTION", 0.003),
        max_motion=_env_float("SWIN_VALLR_VISUAL_FILTER_MAX_MOTION", 0.08),
        min_sharpness=_env_float("SWIN_VALLR_VISUAL_FILTER_MIN_SHARPNESS", 10.0),
        preview_limit=_env_int("SWIN_VALLR_VISUAL_FILTER_PREVIEW_LIMIT", 6),
        preview_frames=_env_int("SWIN_VALLR_VISUAL_FILTER_PREVIEW_FRAMES", 16),
        preview_size=_env_int("SWIN_VALLR_VISUAL_FILTER_PREVIEW_SIZE", 96)
    )


def _safe_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def _sample_frame_indices(total_frames: int, config: VisualFilterConfig) -> List[int]:
    if total_frames <= 0 or config.sample_frames <= 0:
        return []

    stride = max(1, config.sample_stride)
    max_frames = min(config.sample_frames, total_frames)
    indices = []

    for i in range(max_frames):
        idx = i * stride
        if idx >= total_frames:
            break
        indices.append(idx)

    if not indices and total_frames > 0:
        indices = [0]

    return indices


def _score_visual_metrics(metrics: VisualSpeechMetrics, config: VisualFilterConfig) -> float:
    if metrics.frame_samples <= 0:
        return 0.0

    def _ratio(value: float, target: float) -> float:
        if target <= 0:
            return 0.0
        return min(1.0, value / target)

    return (
        0.5 * metrics.face_rate
        + 0.2 * _ratio(metrics.motion_mean, config.min_motion)
        + 0.2 * _ratio(metrics.openness_std, config.min_openness_std)
        + 0.1 * _ratio(metrics.lip_size_ratio, config.min_lip_size_ratio)
    )


def compute_visual_speech_metrics(
    video_path: str,
    processor: MediaPipeProcessor,
    config: VisualFilterConfig
) -> Tuple[VisualSpeechMetrics, List[np.ndarray]]:
    """Compute visual speech quality metrics for a video without audio."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOGGER.warning("Visual prefilter failed to open %s", video_path)
        metrics = VisualSpeechMetrics(
            frame_samples=0,
            detections=0,
            face_rate=0.0,
            avg_lip_size=0.0,
            lip_size_ratio=0.0,
            openness_mean=0.0,
            openness_std=0.0,
            motion_mean=0.0,
            motion_std=0.0,
            sharpness_mean=0.0,
            sharpness_std=0.0,
            score=0.0
        )
        return metrics, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = _sample_frame_indices(total_frames, config)
    frame_samples = len(sample_indices)

    detections = 0
    centers: List[Tuple[float, float]] = []
    sizes: List[float] = []
    openness: List[float] = []
    sharpness: List[float] = []
    preview_frames: List[np.ndarray] = []
    min_dim = 1.0

    progress = tqdm(sample_indices, desc="Prefilter frames", unit="frame", leave=False)
    try:
        for idx in progress:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if frame is None:
                continue

            height, width = frame.shape[:2]
            min_dim = float(min(height, width)) if min(height, width) > 0 else 1.0

            metrics = processor.extract_lip_metrics(frame)
            if metrics is None:
                continue

            detections += 1
            centers.append(metrics.center)
            sizes.append(metrics.size)
            openness.append(metrics.openness)

            roi = _safe_crop(frame, metrics.bbox)
            if roi is not None and roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                if len(preview_frames) < config.preview_frames:
                    preview_frames.append(cv2.resize(roi, (config.preview_size, config.preview_size)))
    finally:
        cap.release()
        progress.close()

    face_rate = float(detections / frame_samples) if frame_samples else 0.0
    avg_lip_size = float(np.mean(sizes)) if sizes else 0.0
    lip_size_ratio = float(avg_lip_size / min_dim) if min_dim > 0 else 0.0

    openness_mean = float(np.mean(openness)) if openness else 0.0
    openness_std = float(np.std(openness)) if openness else 0.0
    openness_mean = float(openness_mean / min_dim) if min_dim > 0 else 0.0
    openness_std = float(openness_std / min_dim) if min_dim > 0 else 0.0

    motion_values: List[float] = []
    previous = None
    for center in centers:
        if previous is not None:
            motion_values.append(float(np.linalg.norm(np.array(center) - np.array(previous))))
        previous = center

    motion_mean = float(np.mean(motion_values)) if motion_values else 0.0
    motion_std = float(np.std(motion_values)) if motion_values else 0.0
    motion_mean = float(motion_mean / min_dim) if min_dim > 0 else 0.0
    motion_std = float(motion_std / min_dim) if min_dim > 0 else 0.0

    sharpness_mean = float(np.mean(sharpness)) if sharpness else 0.0
    sharpness_std = float(np.std(sharpness)) if sharpness else 0.0

    metrics = VisualSpeechMetrics(
        frame_samples=frame_samples,
        detections=detections,
        face_rate=face_rate,
        avg_lip_size=avg_lip_size,
        lip_size_ratio=lip_size_ratio,
        openness_mean=openness_mean,
        openness_std=openness_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        sharpness_mean=sharpness_mean,
        sharpness_std=sharpness_std,
        score=0.0
    )
    metrics.score = _score_visual_metrics(metrics, config)

    return metrics, preview_frames


def passes_visual_filter(
    metrics: VisualSpeechMetrics,
    config: VisualFilterConfig
) -> Tuple[bool, List[str]]:
    """Decide whether a video passes the visual-only prefilter."""
    if metrics.frame_samples <= 0:
        return False, ["no_frames"]

    reasons: List[str] = []
    if metrics.face_rate < config.min_face_rate:
        reasons.append("face_rate")
    if metrics.lip_size_ratio < config.min_lip_size_ratio:
        reasons.append("lip_size")
    if metrics.openness_std < config.min_openness_std:
        reasons.append("openness")
    if metrics.motion_mean < config.min_motion:
        reasons.append("motion_low")
    if config.max_motion > 0 and metrics.motion_mean > config.max_motion:
        reasons.append("motion_high")
    if metrics.sharpness_mean < config.min_sharpness:
        reasons.append("sharpness")

    return len(reasons) == 0, reasons


def _save_visual_filter_artifacts(
    root: Path,
    records: List[Dict],
    rejection_counts: Counter,
    config: VisualFilterConfig
) -> None:
    if not records:
        return

    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "filter_config.json", asdict(config))

    passed = [r for r in records if r.get("passed")]
    rejected = [r for r in records if not r.get("passed")]

    def _avg(rows: List[Dict], key: str) -> float:
        values = [float(r.get(key, 0.0)) for r in rows]
        return float(np.mean(values)) if values else 0.0

    summary = {
        "total_videos": len(records),
        "passed": len(passed),
        "rejected": len(rejected),
        "pass_rate": float(len(passed) / len(records)) if records else 0.0,
        "rejection_reasons": dict(rejection_counts),
        "avg_metrics_passed": {
            "face_rate": _avg(passed, "face_rate"),
            "lip_size_ratio": _avg(passed, "lip_size_ratio"),
            "openness_std": _avg(passed, "openness_std"),
            "motion_mean": _avg(passed, "motion_mean"),
            "sharpness_mean": _avg(passed, "sharpness_mean"),
            "score": _avg(passed, "score")
        },
        "avg_metrics_rejected": {
            "face_rate": _avg(rejected, "face_rate"),
            "lip_size_ratio": _avg(rejected, "lip_size_ratio"),
            "openness_std": _avg(rejected, "openness_std"),
            "motion_mean": _avg(rejected, "motion_mean"),
            "sharpness_mean": _avg(rejected, "sharpness_mean"),
            "score": _avg(rejected, "score")
        }
    }

    save_json(root / "filter_summary.json", summary)

    headers = [
        "id",
        "video",
        "passed",
        "reasons",
        "face_rate",
        "lip_size_ratio",
        "openness_std",
        "motion_mean",
        "motion_std",
        "sharpness_mean",
        "score"
    ]
    rows = []
    for record in records:
        rows.append([
            record.get("id", ""),
            record.get("video", ""),
            record.get("passed", False),
            ";".join(record.get("reasons", [])),
            record.get("face_rate", 0.0),
            record.get("lip_size_ratio", 0.0),
            record.get("openness_std", 0.0),
            record.get("motion_mean", 0.0),
            record.get("motion_std", 0.0),
            record.get("sharpness_mean", 0.0),
            record.get("score", 0.0)
        ])
    save_csv(root / "filter_scores.csv", rows, headers=headers)

    passed_ids = [r.get("id", "") for r in passed]
    rejected_ids = [r.get("id", "") for r in rejected]
    save_text(root / "passed_ids.txt", "\n".join(passed_ids))
    save_text(root / "rejected_ids.txt", "\n".join(rejected_ids))

    series = {
        "score": [r.get("score", 0.0) for r in records],
        "face_rate": [r.get("face_rate", 0.0) for r in records],
        "motion": [r.get("motion_mean", 0.0) for r in records]
    }
    save_line_plot(root / "filter_scores.png", series, title="Visual filter scores")


# =============================================================================
# Language Detection (paper-11+)
# =============================================================================

_ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)]
_CJK_RANGES = [(0x4E00, 0x9FFF)]
_GREEK_RANGES = [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]
_LATIN_EXT_RANGES = [(0x00C0, 0x024F)]

_SPANISH_MARKERS = {
    0x00A1,  # inverted !
    0x00BF,  # inverted ?
    0x00D1,  # N tilde
    0x00F1,  # n tilde
    0x00E1, 0x00E9, 0x00ED, 0x00F3, 0x00FA, 0x00FC  # accents
}

_FRENCH_MARKERS = {
    0x00C0, 0x00C2, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB,
    0x00CE, 0x00CF, 0x00D4, 0x00D9, 0x00DB, 0x00DC, 0x0178,
    0x00E0, 0x00E2, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB,
    0x00EE, 0x00EF, 0x00F4, 0x00F9, 0x00FB, 0x00FC, 0x0153, 0x00E6
}


def _count_in_ranges(text: str, ranges: List[Tuple[int, int]]) -> int:
    count = 0
    for ch in text:
        code = ord(ch)
        for start, end in ranges:
            if start <= code <= end:
                count += 1
                break
    return count


def _count_markers(text: str, markers: set) -> int:
    return sum(1 for ch in text if ord(ch) in markers)


def detect_language(text: str) -> str:
    """Infer a coarse language hint from unicode script."""
    if not text:
        return "unknown"

    arabic = _count_in_ranges(text, _ARABIC_RANGES)
    if arabic > 0:
        return "arabic"

    cjk = _count_in_ranges(text, _CJK_RANGES)
    if cjk > 0:
        return "chinese"

    greek = _count_in_ranges(text, _GREEK_RANGES)
    if greek > 0:
        return "greek"

    latin = 0
    for ch in text:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin += 1
        else:
            code = ord(ch)
            for start, end in _LATIN_EXT_RANGES:
                if start <= code <= end:
                    latin += 1
                    break

    if latin > 0:
        spanish = _count_markers(text, _SPANISH_MARKERS)
        french = _count_markers(text, _FRENCH_MARKERS)
        if spanish > french and spanish > 0:
            return "spanish"
        if french > 0:
            return "french"
        return "english"

    return "unknown"
# =============================================================================
# Dataset Base Class
# =============================================================================

class LipReadingDataset(Dataset):
    """
    Base dataset class for lip reading.

    Expects preprocessed data in the format:
    - videos/: Contains .npy files with lip ROI sequences
    - labels/: Contains .txt files with transcriptions
    - metadata.json: Dataset metadata
    - split.json: (optional) train/val/test split generated by create_split.py
    
    OPTIMIZATION: Supports in-memory caching for faster training.
    """

    def __init__(
        self,
        data_dir: str,
        config: DataConfig = None,
        transform: Optional[Callable] = None,
        phase: CurriculumPhase = CurriculumPhase.FULL_DATASET,
        split: Optional[str] = None,   # "train", "val", "test", or None (all)
        cache_in_memory: bool = True,  # NEW: Cache videos in RAM for speed
    ):
        # CRITICAL FIX: Convert data_dir to absolute path immediately
        self.data_dir = Path(data_dir).resolve()  # resolve() makes it absolute
        self.config = config or DataConfig()
        self.transform = transform
        self.phase = phase
        self.split = split
        self._epoch = 0
        self._training = False  # CRITICAL FIX: Initialize training mode
        self.cache_in_memory = cache_in_memory
        self._video_cache = {}  # NEW: In-memory cache for videos

        self.samples = self._load_samples()
        self.samples = self._filter_by_split(self.samples)
        self.samples = self._filter_by_quality(self.samples)
        self.samples = self._filter_by_phase(self.samples)

        split_tag = f" split={split}" if split else ""
        print(f"[Dataset] Loaded {len(self.samples)} samples (phase: {phase.name}{split_tag})")
        LOGGER.info("Dataset loaded: %s samples (phase=%s, split=%s).",
                    len(self.samples), phase.name, split or "all")
        
        # NEW: Preload videos into memory if caching is enabled
        if self.cache_in_memory:
            self._preload_videos()

    def _preload_videos(self):
        """Preload all videos into memory for faster training."""
        print(f"[Dataset] Preloading {len(self.samples)} videos into memory...")
        LOGGER.info("Preloading %d videos into memory for faster training", len(self.samples))
        
        for sample in tqdm(self.samples, desc="Caching videos", unit="video"):
            video_path = sample.get('video')
            if not os.path.isabs(video_path):
                video_path = str(self.data_dir / video_path)
            
            try:
                # Load video once and cache it
                video = np.load(video_path, allow_pickle=False, mmap_mode=None)
                
                # Validate and normalize shape
                if video.ndim == 3:
                    video = video[..., None]
                if video.ndim != 4:
                    LOGGER.warning("Unexpected video shape %s for %s; using blank sample.", video.shape, video_path)
                    video = np.zeros(
                        (self.config.num_frames, self.config.roi_size, self.config.roi_size, 3),
                        dtype=np.uint8
                    )
                if video.shape[-1] == 1:
                    video = np.repeat(video, 3, axis=-1)
                elif video.shape[-1] > 3:
                    video = video[..., :3]
                
                # Store in cache
                self._video_cache[sample['id']] = video
                
            except Exception as e:
                LOGGER.error(f"Failed to preload video: {video_path} - {e}")
                # Store blank video for failed loads
                self._video_cache[sample['id']] = np.zeros(
                    (self.config.num_frames, self.config.roi_size, self.config.roi_size, 3),
                    dtype=np.uint8
                )
        
        print(f"[Dataset] ✅ Preloaded {len(self._video_cache)} videos into memory")
        LOGGER.info("Preloading complete: %d videos cached", len(self._video_cache))

    def _filter_by_split(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples to the requested split using split.json if present."""
        if self.split is None:
            return samples
        split_path = self.data_dir / "split.json"
        if not split_path.exists():
            LOGGER.warning(
                "split.json not found in %s — using all samples for split=%s. "
                "Run create_split.py to generate a proper split.",
                self.data_dir, self.split
            )
            return samples
        with open(split_path, encoding="utf-8") as f:
            split_data = json.load(f)
        clip_split: dict = split_data.get("clips", {})
        filtered = [s for s in samples if clip_split.get(s["id"]) == self.split]
        LOGGER.info("Split filter '%s': %d / %d samples kept.", self.split, len(filtered), len(samples))
        return filtered

    def _mean_word_confidence(self, sample: Dict) -> float:
        probs = [
            w.get("probability")
            for w in sample.get("words", [])
            if isinstance(w.get("probability"), (int, float))
        ]
        return float(sum(probs) / len(probs)) if probs else 0.0

    def _ctc_required_frames(self, phonemes: List[str]) -> int:
        repeats = sum(1 for a, b in zip(phonemes, phonemes[1:]) if a == b)
        return len(phonemes) + repeats

    def _filter_by_quality(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples that are too noisy or too dense for 50-frame CTC."""
        cfg = self.config
        if (
            cfg.max_words <= 0
            and cfg.max_phoneme_length <= 0
            and cfg.max_ctc_required_frames <= 0
            and cfg.min_word_confidence <= 0
            and cfg.min_face_frame_rate <= 0
            and cfg.min_mouth_motion <= 0
        ):
            return samples

        vocab = set(get_phoneme_vocab())
        vocab.discard("<blank>")
        kept: List[Dict] = []
        rejected = Counter()

        for sample in samples:
            text = sample.get("text", "")
            word_count = int(sample.get("word_count", len(text.split())))
            phonemes = [p for p in text_to_phonemes(text) if p in vocab]
            required_frames = self._ctc_required_frames(phonemes)
            speaker = sample.get("speaker_selection") or {}
            face_rate = float(speaker.get("face_frame_rate", 0.0) or 0.0)
            mouth_motion = float(speaker.get("mouth_motion", 0.0) or 0.0)
            word_confidence = self._mean_word_confidence(sample)

            if cfg.max_words > 0 and word_count > cfg.max_words:
                rejected["too_many_words"] += 1
                continue
            if cfg.max_phoneme_length > 0 and len(phonemes) > cfg.max_phoneme_length:
                rejected["too_many_phonemes"] += 1
                continue
            if cfg.max_ctc_required_frames > 0 and required_frames > cfg.max_ctc_required_frames:
                rejected["ctc_too_dense"] += 1
                continue
            if cfg.min_word_confidence > 0 and word_confidence < cfg.min_word_confidence:
                rejected["low_word_confidence"] += 1
                continue
            if cfg.min_face_frame_rate > 0 and face_rate < cfg.min_face_frame_rate:
                rejected["low_face_rate"] += 1
                continue
            if cfg.min_mouth_motion > 0 and mouth_motion < cfg.min_mouth_motion:
                rejected["low_mouth_motion"] += 1
                continue

            kept.append(sample)

        LOGGER.info(
            "Quality filter kept %d / %d samples. Rejected: %s",
            len(kept), len(samples), dict(rejected)
        )
        print(f"[Dataset] Quality filter kept {len(kept)} / {len(samples)} samples")
        if rejected:
            print(f"[Dataset] Quality filter rejected: {dict(rejected)}")
        return kept

    def _load_samples(self) -> List[Dict]:
        """Load sample metadata."""
        metadata_path = self.data_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding="utf-8") as f:
                    samples = json.load(f)
            except Exception:
                log_exception(LOGGER, "Failed to read metadata.json; falling back to directory scan.")
                return self._scan_samples_from_directories()

            normalized: List[Dict] = []
            for sample in samples:
                if "video" not in sample:
                    LOGGER.warning("Skipping sample without video path: %s", sample)
                    continue
                text = sample.get("text", "")
                if not text:
                    LOGGER.warning("Skipping sample with empty text: %s", sample.get("id", "unknown"))
                    continue
                video_path = Path(sample["video"])
                if not video_path.is_absolute():
                    video_path = self.data_dir / video_path
                if not video_path.exists():
                    LOGGER.warning("Skipping missing video file: %s", video_path)
                    continue
                
                # CRITICAL FIX: Validate file is readable and not corrupted
                try:
                    file_size = video_path.stat().st_size
                    if file_size == 0:
                        LOGGER.warning("Skipping empty video file: %s", video_path)
                        continue
                    # Expected size for 50x96x96x3 uint8 array is ~1.3MB
                    if file_size < 100000:  # Less than 100KB is suspicious
                        LOGGER.warning("Skipping suspiciously small video file (%d bytes): %s", file_size, video_path)
                        continue
                except Exception as e:
                    LOGGER.warning("Skipping unreadable video file: %s - %s", video_path, e)
                    continue
                
                sample["video"] = str(video_path)
                if "word_count" not in sample and "text" in sample:
                    sample["word_count"] = len(text.split())
                if "language" not in sample and "text" in sample:
                    sample["language"] = detect_language(text)
                normalized.append(sample)
            return normalized

        return self._scan_samples_from_directories()

    def _scan_samples_from_directories(self) -> List[Dict]:
        """Scan the dataset folder to build metadata."""
        samples: List[Dict] = []
        video_dir = self.data_dir / "videos"
        label_dir = self.data_dir / "labels"

        if not video_dir.exists():
            LOGGER.warning("Video directory does not exist: %s", video_dir)
            return samples
        if not label_dir.exists():
            LOGGER.warning("Label directory does not exist: %s", label_dir)
            return samples

        video_files = list(video_dir.glob("*.npy"))
        for video_file in tqdm(video_files, desc="Scanning samples", unit="file", leave=False):
            sample_id = video_file.stem
            label_file = label_dir / f"{sample_id}.txt"

            if label_file.exists():
                with open(label_file, 'r') as f:
                    text = f.read().strip()
                if not text:
                    LOGGER.warning("Skipping empty label for %s", sample_id)
                    continue

                samples.append({
                    'id': sample_id,
                    'video': str(video_file),
                    'text': text,
                    'word_count': len(text.split()),
                    'language': detect_language(text)
                })

        return samples

    def _filter_by_phase(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on curriculum phase."""
        if self.phase == CurriculumPhase.SINGLE_WORDS:
            return [s for s in samples if s.get('word_count', 1) == 1]

        if self.phase == CurriculumPhase.SHORT_SENTENCES:
            return [s for s in samples if 1 <= s.get('word_count', 1) <= 5]

        return samples

    def _normalize(self, video: np.ndarray) -> np.ndarray:
        """Normalize video to [-1, 1] range."""
        video = video.astype(np.float32) / 255.0

        mean = np.array(self.config.mean, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array(self.config.std, dtype=np.float32).reshape(1, 1, 1, 3)

        video = (video - mean) / std

        return video

    def _augmentation_scale(self) -> float:
        """Return augmentation strength based on the current epoch."""
        if not self.training:
            return 0.0
        ramp = int(self.config.augmentation_ramp_epochs)
        if ramp <= 0:
            return float(self.config.augmentation_max_strength)
        scale = min(1.0, (self._epoch + 1) / float(ramp))
        return scale * float(self.config.augmentation_max_strength)

    def _maybe_speed_perturb(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply lightweight temporal speed perturbation."""
        if self.config.speed_perturb_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.speed_perturb_prob * scale:
            return video

        factors = self.config.speed_perturb_factors
        if not factors:
            return video

        factor = float(np.random.choice(factors))
        if factor == 1.0 or factor <= 0:
            return video

        num_frames = video.shape[0]
        new_length = max(1, int(round(num_frames / factor)))

        if new_length == num_frames:
            return video

        target_idx = np.linspace(0, num_frames - 1, new_length)
        src_idx = np.clip(np.round(target_idx).astype(np.int64), 0, num_frames - 1)
        resampled = video[src_idx]

        if new_length < num_frames:
            pad_count = num_frames - new_length
            pad_frames = np.repeat(resampled[-1:], pad_count, axis=0)
            return np.concatenate([resampled, pad_frames], axis=0)

        start = np.random.randint(0, new_length - num_frames + 1)
        return resampled[start:start + num_frames]

    def _maybe_temporal_mask(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Randomly mask a short temporal span."""
        if self.config.temporal_mask_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.temporal_mask_prob * scale:
            return video

        max_len = int(round(self.config.temporal_mask_max_len * scale))
        max_len = min(max_len, video.shape[0])
        if max_len <= 0:
            return video

        mask_len = np.random.randint(1, max_len + 1)
        start = np.random.randint(0, video.shape[0] - mask_len + 1)

        masked = video.copy()
        masked[start:start + mask_len] = 0.0
        return masked

    def _maybe_horizontal_flip(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply random horizontal flip."""
        if self.config.horizontal_flip and np.random.random() < 0.5 * scale:
            return video[:, :, ::-1, :].copy()
        return video

    def _maybe_rotate(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply small random rotation to each frame."""
        if self.config.rotation_prob <= 0 or self.config.max_rotation_deg <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.rotation_prob * scale:
            return video

        max_rotation = float(self.config.max_rotation_deg) * scale
        angle = float(np.random.uniform(-max_rotation, max_rotation))
        height, width = video.shape[1:3]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = np.empty_like(video)
        for i, frame in enumerate(video):
            rotated[i] = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

        return rotated

    def _maybe_profile_warp(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply a mild perspective warp to simulate side profiles."""
        if self.config.profile_warp_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.profile_warp_prob * scale:
            return video

        num_frames, height, width, _ = video.shape
        strength = max(0.0, min(0.3, self.config.profile_warp_strength * scale))
        if strength <= 0:
            return video

        shift = int(round(width * strength))
        direction = -1 if np.random.random() < 0.5 else 1
        shift *= direction

        src = np.float32([
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1]
        ])
        dst = np.float32([
            [0 + shift, 0],
            [width - 1 - shift, 0],
            [0, height - 1],
            [width - 1, height - 1]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = np.empty_like(video)
        for i, frame in enumerate(video):
            warped[i] = cv2.warpPerspective(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
        return warped

    def _maybe_spatial_jitter(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply random center/scale jitter to simulate crop variability."""
        if not self.config.random_crop or scale <= 0:
            return video
        if self.config.random_crop_prob <= 0 or np.random.random() >= self.config.random_crop_prob * scale:
            return video

        num_frames, height, width, _ = video.shape
        base_size = min(height, width)

        shrink = np.random.uniform(0.0, self.config.random_crop_scale * scale)
        crop_size = int(round(base_size * (1.0 - shrink)))
        crop_size = max(4, min(crop_size, base_size))

        max_shift = int(round(self.config.random_crop_shift * base_size * scale))
        shift_x = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
        shift_y = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0

        center_x = width // 2 + shift_x
        center_y = height // 2 + shift_y

        x1 = int(round(center_x - crop_size / 2))
        y1 = int(round(center_y - crop_size / 2))
        x1 = max(0, min(x1, width - crop_size))
        y1 = max(0, min(y1, height - crop_size))

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        cropped = video[:, y1:y2, x1:x2, :]
        if cropped.shape[1] == 0 or cropped.shape[2] == 0:
            return video

        resized = np.empty_like(video)
        for i in range(num_frames):
            resized[i] = cv2.resize(cropped[i], (width, height), interpolation=cv2.INTER_LINEAR)

        return resized

    def _apply_color_jitter(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply brightness and contrast jitter."""
        if self.config.brightness_jitter > 0 and scale > 0:
            jitter = self.config.brightness_jitter * scale
            factor = 1 + np.random.uniform(-jitter, jitter)
            video = video * factor

        if self.config.contrast_jitter > 0 and scale > 0:
            jitter = self.config.contrast_jitter * scale
            factor = 1 + np.random.uniform(-jitter, jitter)
            mean = video.mean()
            video = (video - mean) * factor + mean

        return video

    def _maybe_gaussian_blur(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply lightweight Gaussian blur to simulate camera defocus."""
        if self.config.gaussian_blur_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.gaussian_blur_prob * scale:
            return video

        kernels = [k for k in self.config.gaussian_blur_kernel if k >= 3]
        if not kernels:
            return video
        kernel = int(np.random.choice(kernels))
        if kernel % 2 == 0:
            kernel += 1

        blurred = np.empty_like(video)
        for i, frame in enumerate(video):
            blurred[i] = cv2.GaussianBlur(frame, (kernel, kernel), 0)
        return blurred

    def _maybe_gaussian_noise(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply Gaussian noise to simulate sensor noise."""
        if self.config.gaussian_noise_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.gaussian_noise_prob * scale:
            return video

        std = float(self.config.gaussian_noise_std) * scale
        noise = np.random.normal(0.0, std, size=video.shape).astype(np.float32)
        return video + noise

    def _maybe_downscale(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Downscale and upscale to simulate low-resolution captures."""
        if self.config.downscale_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.downscale_prob * scale:
            return video

        min_scale, max_scale = self.config.downscale_range
        min_scale = max(0.1, min(1.0, min_scale))
        max_scale = max(min_scale, min(1.0, max_scale))
        effective_min = 1.0 - (1.0 - min_scale) * scale
        scale_factor = float(np.random.uniform(effective_min, max_scale))
        if scale_factor >= 0.99:
            return video

        num_frames, height, width, _ = video.shape
        new_w = max(1, int(round(width * scale_factor)))
        new_h = max(1, int(round(height * scale_factor)))

        resized = np.empty_like(video)
        for i in range(num_frames):
            small = cv2.resize(video[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized[i] = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized

    def _maybe_random_erasing(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Apply a random rectangular erase to simulate occlusions."""
        if self.config.random_erasing_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.random_erasing_prob * scale:
            return video

        num_frames, height, width, _ = video.shape
        area = height * width
        min_area, max_area = self.config.random_erasing_scale
        min_area = max(0.0, min_area) * scale
        max_area = max(min_area, max_area) * scale
        if max_area <= 0:
            return video

        erase_area = np.random.uniform(min_area, max_area) * area
        min_ratio, max_ratio = self.config.random_erasing_ratio
        aspect_ratio = np.random.uniform(min_ratio, max_ratio)

        erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(erase_area / max(aspect_ratio, 1e-6))))
        if erase_h <= 0 or erase_w <= 0:
            return video
        erase_h = min(erase_h, height)
        erase_w = min(erase_w, width)

        y1 = np.random.randint(0, height - erase_h + 1)
        x1 = np.random.randint(0, width - erase_w + 1)

        erased = video.copy()
        erased[:, y1:y1 + erase_h, x1:x1 + erase_w, :] = self.config.random_erasing_value
        return erased

    def _maybe_beard_occlusion(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Simulate beard/dark lower-face occlusion."""
        if self.config.beard_occlusion_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.beard_occlusion_prob * scale:
            return video

        num_frames, height, width, _ = video.shape
        band_ratio = max(0.05, min(0.7, self.config.beard_occlusion_height_ratio))
        band_height = max(1, int(round(height * band_ratio)))
        start_y = height - band_height

        strength = max(0.0, min(1.0, self.config.beard_occlusion_strength)) * scale
        gradient = np.linspace(0.0, strength, band_height, dtype=np.float32).reshape(band_height, 1, 1)
        noise_level = max(0.0, self.config.beard_occlusion_noise * scale)

        occluded = video.copy()
        for i in range(num_frames):
            region = occluded[i, start_y:height]
            region = region * (1.0 - gradient)
            if noise_level > 0:
                noise = np.random.normal(0.0, noise_level, region.shape).astype(np.float32)
                region = region + noise
            occluded[i, start_y:height] = region

        return occluded

    def _maybe_black_bar(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Add a dark horizontal bar occlusion (mask/black log)."""
        if self.config.black_bar_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.black_bar_prob * scale:
            return video

        num_frames, height, width, _ = video.shape
        bar_ratio = max(0.05, min(0.6, self.config.black_bar_height_ratio))
        bar_height = max(1, int(round(height * bar_ratio)))
        start_y = int(round((height - bar_height) * 0.5))

        occluded = video.copy()
        occluded[:, start_y:start_y + bar_height, :, :] = self.config.black_bar_value
        return occluded

    def _maybe_frame_dropout(self, video: np.ndarray, scale: float) -> np.ndarray:
        """Randomly replace a few frames to simulate dropped frames."""
        if self.config.frame_dropout_prob <= 0 or scale <= 0:
            return video
        if np.random.random() >= self.config.frame_dropout_prob * scale:
            return video

        num_frames = video.shape[0]
        max_ratio = max(0.0, self.config.frame_dropout_max_ratio * scale)
        drop_count = int(round(num_frames * np.random.uniform(0.0, max_ratio)))
        if drop_count <= 0 or num_frames <= 1:
            return video

        drop_indices = np.random.choice(num_frames, size=drop_count, replace=False)
        replaced = video.copy()
        for idx in drop_indices:
            if idx > 0:
                replaced[idx] = replaced[idx - 1]
            else:
                replaced[idx] = replaced[min(1, num_frames - 1)]
        return replaced

    def _maybe_spec_augment(self, video: np.ndarray, scale: float) -> np.ndarray:
        """SpecAugment-style frequency and time masking for visual speech."""
        if scale <= 0 or np.random.random() >= 0.3 * scale:
            return video
        
        # Time masking (mask consecutive frames)
        if np.random.random() < 0.5:
            mask_len = np.random.randint(1, min(8, max(2, video.shape[0] // 4)))
            start = np.random.randint(0, max(1, video.shape[0] - mask_len + 1))
            video[start:start + mask_len] = 0
        
        # Spatial masking (mask patches in mouth region)
        if np.random.random() < 0.5:
            h, w = video.shape[1:3]
            mask_h = np.random.randint(max(1, h // 8), max(2, h // 4))
            mask_w = np.random.randint(max(1, w // 8), max(2, w // 4))
            start_h = np.random.randint(0, max(1, h - mask_h + 1))
            start_w = np.random.randint(0, max(1, w - mask_w + 1))
            video[:, start_h:start_h + mask_h, start_w:start_w + mask_w] = 0
        
        return video

    def _augment(self, video: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        if not self.training:
            return video

        scale = self._augmentation_scale()
        video = self._maybe_speed_perturb(video, scale)
        video = self._maybe_frame_dropout(video, scale)
        video = self._maybe_spatial_jitter(video, scale)
        video = self._maybe_horizontal_flip(video, scale)
        video = self._maybe_rotate(video, scale)
        video = self._maybe_profile_warp(video, scale)
        video = self._apply_color_jitter(video, scale)
        video = self._maybe_gaussian_blur(video, scale)
        video = self._maybe_downscale(video, scale)
        video = self._maybe_gaussian_noise(video, scale)
        video = self._maybe_random_erasing(video, scale)
        video = self._maybe_beard_occlusion(video, scale)
        video = self._maybe_black_bar(video, scale)
        video = self._maybe_temporal_mask(video, scale)
        video = self._maybe_spec_augment(video, scale)  # NEW: SpecAugment

        return np.clip(video, -3, 3)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sample_id = sample['id']
        
        # OPTIMIZATION: Use cached video if available
        if self.cache_in_memory and sample_id in self._video_cache:
            video = self._video_cache[sample_id].copy()  # Copy to avoid modifying cache
        else:
            # Fallback to disk loading if not cached
            video_path = sample.get('video')
            
            # Path should already be absolute from _load_samples, but double-check
            if not os.path.isabs(video_path):
                video_path = str(self.data_dir / video_path)
            
            try:
                # CRITICAL FIX: Use allow_pickle=False and mmap_mode=None for safety
                video = np.load(video_path, allow_pickle=False, mmap_mode=None)
            except Exception as e:
                LOGGER.error(f"Failed to load video sample: {video_path} - {e}")
                # Return a blank video instead of crashing
                video = np.zeros(
                    (self.config.num_frames, self.config.roi_size, self.config.roi_size, 3),
                    dtype=np.uint8
                )

            if video.ndim == 3:
                video = video[..., None]
            if video.ndim != 4:
                LOGGER.warning("Unexpected video shape %s for %s; using blank sample.", video.shape, video_path)
                video = np.zeros(
                    (self.config.num_frames, self.config.roi_size, self.config.roi_size, 3),
                    dtype=np.uint8
                )
            if video.shape[-1] == 1:
                video = np.repeat(video, 3, axis=-1)
            elif video.shape[-1] > 3:
                video = video[..., :3]

        # Apply normalization and augmentation
        video = self._normalize(video)
        video = self._augment(video)

        video = torch.from_numpy(video).permute(3, 0, 1, 2).float()
        text = sample.get('text', "")

        return {
            'video': video,
            'text': text,
            'id': sample_id
        }

    @property
    def training(self) -> bool:
        """Check if in training mode."""
        return hasattr(self, '_training') and self._training

    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        return self

    def set_epoch(self, epoch: int):
        """Set epoch for augmentation scheduling."""
        self._epoch = max(0, int(epoch))
        return self


# =============================================================================
# Curriculum Learning Sampler
# =============================================================================

class CurriculumSampler(Sampler):
    """
    Sampler that implements curriculum learning by epoch.

    Phase 1 (epochs 1-5): Single words
    Phase 2 (epochs 6-20): Short sentences
    Phase 3 (epochs 21+): Full dataset
    """

    def __init__(
        self,
        dataset: LipReadingDataset,
        epoch: int = 0,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.epoch = epoch
        self.shuffle = shuffle

        self.single_word_indices: List[int] = []
        self.short_sentence_indices: List[int] = []
        self.all_indices = list(range(len(dataset)))

        for i, sample in enumerate(dataset.samples):
            word_count = sample.get('word_count', 1)
            if word_count == 1:
                self.single_word_indices.append(i)
            if 1 <= word_count <= 5:
                self.short_sentence_indices.append(i)

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self.epoch = epoch

    def _get_phase_indices(self) -> List[int]:
        """Get indices for current curriculum phase."""
        if self.epoch < 5:
            return self.single_word_indices if self.single_word_indices else self.all_indices
        if self.epoch < 20:
            return self.short_sentence_indices if self.short_sentence_indices else self.all_indices
        return self.all_indices

    def __iter__(self):
        indices = self._get_phase_indices()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        return iter(indices)

    def __len__(self) -> int:
        return len(self._get_phase_indices())


# =============================================================================
# Language-Balanced Sampler (paper-11+)
# =============================================================================

class LanguageBalancedSampler(Sampler):
    """Sample examples with inverse-frequency language weights."""

    def __init__(
        self,
        dataset: LipReadingDataset,
        epoch: int = 0,
        power: float = 0.7,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.epoch = epoch
        self.power = max(0.0, float(power))
        self.shuffle = shuffle
        self.weights = self._build_weights()

    def _build_weights(self) -> torch.Tensor:
        counts = Counter(sample.get("language", "unknown") for sample in self.dataset.samples)
        weights = []
        for sample in self.dataset.samples:
            lang = sample.get("language", "unknown")
            count = max(1, counts.get(lang, 1))
            weight = (1.0 / float(count)) ** self.power
            weights.append(weight)
        if not weights:
            return torch.zeros(0, dtype=torch.double)
        return torch.tensor(weights, dtype=torch.double)

    def set_epoch(self, epoch: int):
        """Update current epoch for deterministic sampling."""
        self.epoch = epoch

    def __iter__(self):
        if self.weights.numel() == 0:
            return iter([])
        num_samples = len(self.dataset)
        generator = torch.Generator()
        if self.shuffle:
            generator.manual_seed(self.epoch)
        indices = torch.multinomial(
            self.weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator
        ).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# Dataset Download & Setup
# =============================================================================

def download_and_mix_datasets():
    """
    Interactive function to download and prepare lip reading datasets.

    Supports:
    - LRS2 (BBC)
    - LRS3 (TED Talks)
    - Custom datasets
    """
    print("\n" + "="*60)
    print("SWIN-VALLR Dataset Setup")
    print("="*60)
    LOGGER.info("Starting dataset setup.")

    save_dir = input("\nPlease enter the path to save/load datasets: ").strip()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Setup] Using directory: {save_dir}")
    LOGGER.info("Using dataset directory: %s", save_dir)

    if (save_dir / "metadata.json").exists():
        response = input("\nExisting preprocessed data found. Use it? [Y/n]: ").strip().lower()
        if response != 'n':
            print("[Setup] Using existing preprocessed data.")
            LOGGER.info("Using existing preprocessed data.")
            return str(save_dir)

    print("\nAvailable datasets:")
    print("  1. LRS2 (BBC - download manually)")
    print("  2. LRS3 (TED Talks - download manually)")
    print("  3. Use custom video directory")
    print("  4. Download sample dataset (for testing)")

    choice = input("\nSelect dataset [1-4]: ").strip()
    LOGGER.info("Dataset selection: %s", choice)

    if choice == '4':
        print("\n[Setup] Downloading sample dataset...")
        LOGGER.info("Downloading sample dataset.")
        _download_sample_dataset(save_dir)

    elif choice == '3':
        video_dir = input("\nEnter path to video directory: ").strip()
        label_dir = input("Enter path to label directory (or press Enter to skip): ").strip()

        print("\n[Setup] Processing custom dataset...")
        LOGGER.info("Processing custom dataset: videos=%s labels=%s", video_dir, label_dir)
        _process_custom_dataset(save_dir, video_dir, label_dir or None)

    else:
        dataset_name = "LRS2" if choice == '1' else "LRS3"
        print(f"\n[Setup] {dataset_name} requires manual download due to licensing.")
        print(f"Please download from: https://www.robots.ox.ac.uk/~vgg/data/{dataset_name.lower()}/")

        zip_path = input(f"\nEnter path to downloaded {dataset_name} zip file: ").strip()

        if zip_path and Path(zip_path).exists():
            print(f"\n[Setup] Processing {dataset_name}...")
            LOGGER.info("Processing %s from zip: %s", dataset_name, zip_path)
            _process_lrs_dataset(save_dir, zip_path, dataset_name)
        else:
            print("[Setup] File not found. Please download and run setup again.")
            LOGGER.error("Dataset zip not found: %s", zip_path)
            return None

    print(f"\n[Setup] Dataset prepared at: {save_dir}")
    LOGGER.info("Dataset prepared at: %s", save_dir)
    return str(save_dir)


def _download_sample_dataset(save_dir: Path):
    """Download a small sample dataset for testing."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        print("[Setup] Attempting to download from HuggingFace Hub...")
        LOGGER.info("Attempting to download sample dataset via huggingface_hub.")

        _create_placeholder_data(save_dir)

    except ImportError:
        print("[Setup] huggingface_hub not available, creating placeholder data...")
        LOGGER.warning("huggingface_hub not available; creating placeholder data.")
        _create_placeholder_data(save_dir)
    except Exception as e:
        log_exception(LOGGER, "Sample dataset download failed.")
        print(f"[Setup] Download failed: {e}")
        print("[Setup] Creating placeholder data for testing...")
        _create_placeholder_data(save_dir)


def _create_placeholder_data(save_dir: Path):
    """Create placeholder data for testing the pipeline."""
    video_dir = save_dir / "videos"
    label_dir = save_dir / "labels"
    video_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    samples = [
        ("hello", "hello"),
        ("world", "world"),
        ("how_are_you", "how are you"),
        ("thank_you", "thank you"),
        ("goodbye", "goodbye"),
    ]

    metadata: List[Dict] = []

    for sample_id, text in tqdm(samples, desc="Creating placeholder samples", unit="sample", leave=False):
        video = np.random.randint(0, 255, (DEFAULT_NUM_FRAMES, DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE, 3), dtype=np.uint8)

        np.save(video_dir / f"{sample_id}.npy", video)
        _save_roi_preview(sample_id, video)

        with open(label_dir / f"{sample_id}.txt", 'w') as f:
            f.write(text)

        metadata.append({
            'id': sample_id,
            'video': str(video_dir / f"{sample_id}.npy"),
            'text': text,
            'word_count': len(text.split())
        })

    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Setup] Created {len(samples)} placeholder samples for testing.")
    LOGGER.info("Created %d placeholder samples.", len(samples))
    _save_dataset_artifacts(save_dir, metadata, "placeholder")


def _process_custom_dataset(
    save_dir: Path,
    video_dir: str,
    label_dir: Optional[str],
    filter_config: Optional[VisualFilterConfig] = None
):
    """Process a custom video directory."""
    video_path = Path(video_dir)
    label_path = Path(label_dir) if label_dir else None

    out_video_dir = save_dir / "videos"
    out_label_dir = save_dir / "labels"
    out_video_dir.mkdir(exist_ok=True)
    out_label_dir.mkdir(exist_ok=True)

    if not MEDIAPIPE_AVAILABLE:
        print("[Setup] MediaPipe not available - cannot process videos")
        LOGGER.error("MediaPipe unavailable; cannot process custom dataset.")
        return

    processor = MediaPipeProcessor(roi_size=DEFAULT_ROI_SIZE)
    metadata: List[Dict] = []
    filter_config = filter_config or _visual_filter_config_from_env()
    filter_records: List[Dict] = []
    rejection_counts: Counter = Counter()
    filter_root = None
    preview_budget = {"accepted": 0, "rejected": 0}

    if filter_config.enabled:
        filter_root = get_useful_dir("data_pipeline", "visual_filter")
        (filter_root / "previews" / "accepted").mkdir(parents=True, exist_ok=True)
        (filter_root / "previews" / "rejected").mkdir(parents=True, exist_ok=True)
        preview_budget = {
            "accepted": filter_config.preview_limit,
            "rejected": filter_config.preview_limit
        }
        LOGGER.info("Visual prefilter enabled: %s", asdict(filter_config))

    video_files = [f for f in video_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

    print(f"[Setup] Processing {len(video_files)} videos...")
    LOGGER.info("Processing %d videos from %s", len(video_files), video_path)

    for video_file in tqdm(video_files, desc="Processing"):
        sample_id = video_file.stem
        visual_metrics = None

        if filter_config.enabled:
            visual_metrics, preview_frames = compute_visual_speech_metrics(
                str(video_file),
                processor,
                filter_config
            )
            passed, reasons = passes_visual_filter(visual_metrics, filter_config)
            record = {
                "id": sample_id,
                "video": str(video_file),
                "passed": passed,
                "reasons": reasons
            }
            record.update(asdict(visual_metrics))
            filter_records.append(record)

            status = "accepted" if passed else "rejected"
            if filter_root and preview_frames and preview_budget.get(status, 0) > 0:
                preview_dir = filter_root / "previews" / status
                save_montage(
                    preview_dir / f"{sample_id}.png",
                    preview_frames,
                    cols=8,
                    pad=2
                )
                preview_budget[status] -= 1

            if not passed:
                for reason in reasons:
                    rejection_counts[reason] += 1
                LOGGER.info("Skipping %s (visual filter: %s)", sample_id, ",".join(reasons))
                continue

        rois = processor.process_video(str(video_file), target_fps=DEFAULT_TARGET_FPS, num_frames=DEFAULT_NUM_FRAMES)

        if rois is None:
            print(f"  Skipping {sample_id} - processing failed")
            LOGGER.warning("Skipping %s - ROI extraction failed.", sample_id)
            continue

        np.save(out_video_dir / f"{sample_id}.npy", rois)
        _save_roi_preview(sample_id, rois)

        text = ""
        if label_path and (label_path / f"{sample_id}.txt").exists():
            with open(label_path / f"{sample_id}.txt", 'r') as f:
                text = f.read().strip()

        with open(out_label_dir / f"{sample_id}.txt", 'w') as f:
            f.write(text)

        entry = {
            'id': sample_id,
            'video': str(out_video_dir / f"{sample_id}.npy"),
            'text': text,
            'word_count': len(text.split()) if text else 0
        }
        if visual_metrics is not None:
            entry['quality_metrics'] = asdict(visual_metrics)
        metadata.append(entry)

    processor.close()

    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Setup] Processed {len(metadata)} videos.")
    LOGGER.info("Processed %d videos into dataset.", len(metadata))
    _save_dataset_artifacts(save_dir, metadata, "custom")

    if filter_config.enabled and filter_root is not None:
        _save_visual_filter_artifacts(filter_root, filter_records, rejection_counts, filter_config)


def _process_lrs_dataset(save_dir: Path, zip_path: str, dataset_name: str):
    """Process LRS2 or LRS3 dataset from zip file."""
    import zipfile

    print(f"[Setup] Extracting {dataset_name}...")
    LOGGER.info("Extracting %s from %s", dataset_name, zip_path)

    extract_dir = save_dir / f"{dataset_name}_raw"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"[Setup] Processing extracted files...")

    video_files = list(extract_dir.rglob("*.mp4"))

    if not video_files:
        print("[Setup] No video files found in archive.")
        LOGGER.warning("No video files found in %s archive.", extract_dir)
        return

    _process_custom_dataset(
        save_dir,
        str(video_files[0].parent),
        None
    )


def _save_dataset_artifacts(save_dir: Path, metadata: List[Dict], dataset_type: str) -> None:
    """Save dataset summary artifacts for research."""
    try:
        root = get_useful_dir("data_pipeline", "dataset_summary")
        word_counts = [m.get("word_count", 0) for m in metadata]
        distribution: Dict[str, int] = {}
        for count in word_counts:
            key = str(count)
            distribution[key] = distribution.get(key, 0) + 1
        language_counts = Counter(m.get("language", "unknown") for m in metadata)
        summary = {
            "dataset_type": dataset_type,
            "save_dir": str(save_dir),
            "num_samples": len(metadata),
            "avg_words": float(np.mean(word_counts)) if metadata else 0.0,
            "word_count_distribution": distribution,
            "language_distribution": dict(language_counts)
        }
        save_json(root / "summary.json", summary)
        save_json(root / "metadata.json", metadata)
        save_text(root / "notes.txt", "Dataset summary and metadata snapshot.")
        LOGGER.info("Saved dataset artifacts to %s", root)
    except Exception:
        log_exception(LOGGER, "Failed to save dataset artifacts.")


def _save_roi_preview(sample_id: str, rois: np.ndarray) -> None:
    """Save a montage preview for a sample's ROI sequence."""
    try:
        root = get_useful_dir("data_pipeline", "roi_previews")
        frames = [frame for frame in rois[:min(len(rois), 20)]]
        save_montage(root / f"{sample_id}.png", frames, cols=10, pad=2)
    except Exception:
        log_exception(LOGGER, f"Failed to save ROI preview for {sample_id}.")


# =============================================================================
# Data Loader Factory
# =============================================================================

def create_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    epoch: int = 0,
    training: bool = True,
    config: DataConfig = None,
    split: Optional[str] = None,   # "train", "val", "test", or None (legacy all)
    cache_in_memory: bool = True,  # NEW: Cache videos in RAM for speed
) -> DataLoader:
    """
    Create a DataLoader with curriculum learning support.

    Args:
        data_dir: Path to preprocessed dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        epoch: Current epoch (for curriculum learning)
        training: Whether in training mode (controls augmentation/shuffle)
        config: Data configuration
        split: Which split to load ("train", "val", "test", or None for all).
               Requires split.json generated by create_split.py.
        cache_in_memory: Whether to preload all videos into RAM (much faster!)

    Returns:
        PyTorch DataLoader
    """
    config = config or DataConfig()
    if _env_flag("SWIN_VALLR_LANGUAGE_BALANCE"):
        config.language_balance = True
    config.language_balance_power = _env_float(
        "SWIN_VALLR_LANGUAGE_BALANCE_POWER",
        config.language_balance_power
    )

    if epoch < config.curriculum_phases[0]:
        phase = CurriculumPhase.SINGLE_WORDS
    elif epoch < config.curriculum_phases[1]:
        phase = CurriculumPhase.SHORT_SENTENCES
    else:
        phase = CurriculumPhase.FULL_DATASET

    dataset = LipReadingDataset(
        data_dir, 
        config=config, 
        phase=phase, 
        split=split,
        cache_in_memory=cache_in_memory  # NEW: Enable caching
    )
    dataset.train(training)
    dataset.set_epoch(epoch)
    LOGGER.info(
        "Creating dataloader: data_dir=%s batch_size=%s epoch=%s training=%s phase=%s split=%s cache=%s",
        data_dir, batch_size, epoch, training, phase.name, split or "all", cache_in_memory
    )

    if training:
        if config.language_balance:
            sampler = LanguageBalancedSampler(
                dataset,
                epoch=epoch,
                power=config.language_balance_power,
                shuffle=True
            )
        elif max(config.curriculum_phases) <= 0:
            sampler = None
        else:
            sampler = CurriculumSampler(dataset, epoch=epoch, shuffle=True)
    else:
        sampler = None

    # OPTIMIZATION: When caching is enabled, use fewer workers since data is in RAM
    effective_workers = 0 if cache_in_memory else num_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(training and sampler is None),
        num_workers=effective_workers,  # 0 workers when cached (faster!)
        pin_memory=True,
        drop_last=training,
        collate_fn=_collate_fn,
        # CRITICAL FIX: Prevent hanging with multiprocessing
        persistent_workers=(effective_workers > 0),  # Keep workers alive between epochs
        prefetch_factor=2 if effective_workers > 0 else None,  # Prefetch 2 batches per worker
        timeout=30 if effective_workers > 0 else 0,  # 30 second timeout for worker processes
    )

    return dataloader


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length text."""
    videos = torch.stack([item['video'] for item in batch])
    texts = [item['text'] for item in batch]
    ids = [item['id'] for item in batch]
    languages = [item.get('language', 'unknown') for item in batch]

    return {
        'video': videos,
        'text': texts,
        'id': ids,
        'language': languages
    }


# =============================================================================
# Phoneme Utilities
# =============================================================================

# Basic grapheme-to-phoneme mapping (simplified)
# In production, use a proper G2P library like g2p_en
GRAPHEME_TO_PHONEME = {
    'a': ['AH'], 'e': ['EH'], 'i': ['IH'], 'o': ['AA'], 'u': ['AH'],
    'b': ['B'], 'c': ['K'], 'd': ['D'], 'f': ['F'], 'g': ['G'],
    'h': ['HH'], 'j': ['JH'], 'k': ['K'], 'l': ['L'], 'm': ['M'],
    'n': ['N'], 'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'],
    't': ['T'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'],
    'z': ['Z']
}


def text_to_phonemes(text: str) -> List[str]:
    """
    Convert text to phoneme sequence (simplified).

    For production, use g2p_en or similar library.
    """
    vocab = set(get_phoneme_vocab())
    vocab.discard('<blank>')
    phonemes: List[str] = []
    for char in text.lower():
        if char in GRAPHEME_TO_PHONEME:
            phonemes.extend(p for p in GRAPHEME_TO_PHONEME[char] if p in vocab)
    return phonemes


def get_phoneme_vocab() -> List[str]:
    """Get the phoneme vocabulary."""
    return [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '<blank>'
    ]


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    LOGGER.info("Running data_pipeline entrypoint.")
    log_system_info(LOGGER)
    data_dir = download_and_mix_datasets()

    if data_dir:
        print("\n[Test] Creating dataloader...")
        dataloader = create_dataloader(data_dir, batch_size=2, num_workers=0, epoch=0)

        print(f"[Test] Dataloader created with {len(dataloader)} batches")

        for batch in dataloader:
            print(f"\n[Test] Sample batch:")
            print(f"  Video shape: {batch['video'].shape}")
            print(f"  Texts: {batch['text']}")
            print(f"  IDs: {batch['id']}")
            break
