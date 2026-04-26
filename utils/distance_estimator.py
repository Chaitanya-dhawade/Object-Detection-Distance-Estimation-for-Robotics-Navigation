"""
utils/distance_estimator.py
────────────────────────────
Monocular distance estimation using the pinhole camera model.

Formula:
    Distance (m) = (Real_Height_m × Focal_Length_px) / Pixel_Height_px

The focal length in pixels can be obtained from camera calibration or
estimated from a known object at a known distance.

Reference:
    Pinhole camera model: https://en.wikipedia.org/wiki/Pinhole_camera_model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Real-world object heights (metres) ─────────────────────────────────────
DEFAULT_REAL_HEIGHTS: Dict[str, float] = {
    "cone":      0.70,   # traffic cone
    "barrier":   1.00,   # jersey / water barrier
    "stop_sign": 0.75,   # standard stop sign panel
}

# ─── Default KITTI camera focal length ──────────────────────────────────────
DEFAULT_FOCAL_LENGTH_PX = 718.856   # pixels  (KITTI cam2)


@dataclass
class Detection:
    """Single detection result enriched with distance."""
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  Tuple[int, int, int, int]    # (x1, y1, x2, y2) in pixels
    distance_m: Optional[float] = None        # estimated distance, metres

    @property
    def label(self) -> str:
        if self.distance_m is not None:
            return f"{self.class_name}, {self.distance_m:.1f}m"
        return self.class_name


# ─── Estimator ───────────────────────────────────────────────────────────────

class DistanceEstimator:
    """
    Estimates the distance to a detected object using the pinhole
    camera model and a known real-world object height.

    Args:
        focal_length_px:  Camera focal length in pixels.
                          Calibrate with: f = (D * P) / H
                          where D = known distance, P = pixel height at D,
                          H = real height.
        real_heights:     Dict mapping class_name → real height in metres.
    """

    def __init__(
        self,
        focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
        real_heights: Dict[str, float] = None,
    ):
        self.focal_length_px = focal_length_px
        self.real_heights: Dict[str, float] = (
            real_heights if real_heights is not None else DEFAULT_REAL_HEIGHTS.copy()
        )
        logger.debug(
            f"DistanceEstimator initialised – f={focal_length_px:.1f}px, "
            f"classes={list(self.real_heights)}"
        )

    # ── Core formula ─────────────────────────────────────────────────────────

    def estimate(self, class_name: str, pixel_height: float) -> Optional[float]:
        """
        Compute distance in metres.

        Args:
            class_name:    Detected object class (must be in real_heights).
            pixel_height:  Bounding-box height in pixels.

        Returns:
            Distance in metres, or None if class is unknown / bbox is degenerate.
        """
        if pixel_height <= 0:
            return None

        real_h = self.real_heights.get(class_name)
        if real_h is None:
            logger.debug(f"No real height for class '{class_name}' – skipping distance")
            return None

        distance = (real_h * self.focal_length_px) / pixel_height
        return round(distance, 2)

    def estimate_from_bbox(
        self, class_name: str, bbox_xyxy: Tuple[int, int, int, int]
    ) -> Optional[float]:
        """
        Convenience wrapper: compute pixel height from (x1,y1,x2,y2) bbox.
        """
        x1, y1, x2, y2 = bbox_xyxy
        pixel_height = max(y2 - y1, 1)
        return self.estimate(class_name, pixel_height)

    def enrich_detections(self, detections: list[Detection]) -> list[Detection]:
        """
        In-place: add `.distance_m` to every Detection in the list.
        Returns the same list for chaining.
        """
        for det in detections:
            det.distance_m = self.estimate_from_bbox(det.class_name, det.bbox_xyxy)
        return detections

    # ── Calibration helper ───────────────────────────────────────────────────

    @staticmethod
    def calibrate_focal_length(
        known_distance_m: float,
        real_height_m:    float,
        pixel_height:     float,
    ) -> float:
        """
        Compute focal length from a calibration image.

        Place an object of known height at a known distance from the camera.
        Measure its pixel height in the image.

        Args:
            known_distance_m:  Distance to calibration object (metres).
            real_height_m:     Real height of calibration object (metres).
            pixel_height:      Pixel height in the calibration image.

        Returns:
            Focal length in pixels.
        """
        focal = (known_distance_m * pixel_height) / real_height_m
        logger.info(f"Calibrated focal length: {focal:.2f} px")
        return focal


# ─── Annotation helper ───────────────────────────────────────────────────────

class BBoxAnnotator:
    """
    Draws bounding boxes + distance labels onto an OpenCV frame.

    Colour scheme:
        cone      → orange
        barrier   → red
        stop_sign → green
    """

    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "cone":      (0, 165, 255),   # orange  (BGR)
        "barrier":   (0,  50, 255),   # red
        "stop_sign": (0, 200,  60),   # green
    }
    DEFAULT_COLOR = (200, 200, 200)   # grey fallback

    def __init__(self, font_scale: float = 0.65, thickness: int = 2):
        self.font_scale = font_scale
        self.thickness  = thickness
        self.font       = cv2.FONT_HERSHEY_DUPLEX

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw all detections on *frame* (in-place).

        Returns the annotated frame.
        """
        for det in detections:
            color = self.CLASS_COLORS.get(det.class_name, self.DEFAULT_COLOR)
            x1, y1, x2, y2 = det.bbox_xyxy

            # ── Bounding box ─────────────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            # ── Label background ─────────────────────────────────────────
            label     = det.label
            (tw, th), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            bg_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (x1, bg_y1),
                (x1 + tw + 4, y1),
                color, -1,   # filled
            )

            # ── Label text ───────────────────────────────────────────────
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - baseline - 2),
                self.font, self.font_scale,
                (255, 255, 255),    # white text
                self.thickness, cv2.LINE_AA,
            )

            # ── Confidence (small) ───────────────────────────────────────
            conf_txt = f"{det.confidence:.2f}"
            cv2.putText(
                frame, conf_txt,
                (x1 + 2, y2 - 4),
                cv2.FONT_HERSHEY_PLAIN, 0.9,
                color, 1, cv2.LINE_AA,
            )

        return frame
