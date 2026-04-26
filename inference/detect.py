"""
inference/detect.py
────────────────────
Production inference pipeline.

Pipeline:
  Image / Video / Webcam
      ↓
  YOLOv8 detection (GPU or CPU)
      ↓
  Distance estimation (pinhole model)
      ↓
  Annotated output (display / save)

Usage:
    # Image
    python inference/detect.py --source image.jpg --weights runs/train/best.pt

    # Video file
    python inference/detect.py --source video.mp4 --weights best.pt --save

    # Webcam
    python inference/detect.py --source 0 --weights best.pt

    # Use optimised (quantised) ONNX model
    python inference/detect.py --source image.jpg --weights model_int8.onnx
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# Project-level imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.distance_estimator import (
    BBoxAnnotator,
    Detection,
    DistanceEstimator,
    DEFAULT_REAL_HEIGHTS,
)

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Class names (must match training order) ─────────────────────────────────
CLASS_NAMES = ["cone", "barrier", "stop_sign"]


# ─── Model wrapper ───────────────────────────────────────────────────────────

class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO for our 3-class task.

    Supports .pt (PyTorch) and .onnx (via onnxruntime) weights.
    """

    def __init__(
        self,
        weights:    str,
        conf_thres: float = 0.40,
        iou_thres:  float = 0.50,
        device:     str   = "cuda",
        img_size:   int   = 640,
    ):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.img_size   = img_size
        self.device     = self._resolve_device(device)

        suffix = Path(weights).suffix.lower()
        if suffix in (".pt", ".pth", ""):
            self._load_ultralytics(weights)
        elif suffix == ".onnx":
            self._load_onnx(weights)
        else:
            raise ValueError(f"Unsupported weight format: {suffix}")

        logger.info(f"Model loaded: {weights}  |  device: {self.device}")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_ultralytics(self, weights: str) -> None:
        from ultralytics import YOLO
        self.model  = YOLO(weights)
        self._mode  = "ultralytics"

    def _load_onnx(self, weights: str) -> None:
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self._mode = "onnx"
        logger.info(f"ONNX providers: {self.session.get_providers()}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Returns:
            List of Detection objects (unsorted).
        """
        if self._mode == "ultralytics":
            return self._predict_ultralytics(frame)
        return self._predict_onnx(frame)

    def _predict_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf    = self.conf_thres,
            iou     = self.iou_thres,
            imgsz   = self.img_size,
            device  = self.device,
            verbose = False,
        )
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    class_id   = cls_id,
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                    confidence = conf,
                    bbox_xyxy  = (x1, y1, x2, y2),
                ))
        return detections

    def _predict_onnx(self, frame: np.ndarray) -> List[Detection]:
        """ONNX Runtime inference path."""
        blob = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess_onnx(outputs, frame.shape)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR → normalised CHW float32 blob."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]    # (1,3,H,W)
        return np.ascontiguousarray(img)

    def _postprocess_onnx(
        self,
        outputs: list,
        orig_shape: Tuple[int, int, int],
    ) -> List[Detection]:
        """
        Parse YOLOv8 ONNX output [1, 84, N] or [1, N, 84].
        Applies conf + NMS filtering.
        """
        pred = outputs[0]
        if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))   # → [1, N, 84]

        pred = pred[0]                               # [N, 84]
        boxes_cxcywh = pred[:, :4]
        scores       = pred[:, 4:]
        class_ids    = np.argmax(scores, axis=1)
        confidences  = scores[np.arange(len(scores)), class_ids]

        # Confidence filter
        mask = confidences >= self.conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences  = confidences[mask]
        class_ids    = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2 (normalised → pixel)
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size

        detections: List[Detection] = []
        # NMS per class
        for cls_id in np.unique(class_ids):
            idxs = np.where(class_ids == cls_id)[0]
            cls_boxes = boxes_cxcywh[idxs]
            cls_confs = confidences[idxs]

            # cxcywh → xyxy in pixel coords
            x1s = (cls_boxes[:, 0] - cls_boxes[:, 2] / 2) * scale_x
            y1s = (cls_boxes[:, 1] - cls_boxes[:, 3] / 2) * scale_y
            x2s = (cls_boxes[:, 0] + cls_boxes[:, 2] / 2) * scale_x
            y2s = (cls_boxes[:, 1] + cls_boxes[:, 3] / 2) * scale_y

            rects   = np.stack([x1s, y1s, x2s - x1s, y2s - y1s], axis=1)
            scores  = cls_confs.tolist()
            nms_idx = cv2.dnn.NMSBoxes(
                rects.tolist(), scores, self.conf_thres, self.iou_thres
            )
            if nms_idx is None:
                continue
            nms_idx = nms_idx.flatten()

            for i in nms_idx:
                x1 = int(np.clip(x1s[i], 0, orig_w))
                y1 = int(np.clip(y1s[i], 0, orig_h))
                x2 = int(np.clip(x2s[i], 0, orig_w))
                y2 = int(np.clip(y2s[i], 0, orig_h))
                detections.append(Detection(
                    class_id   = int(cls_id),
                    class_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(cls_id),
                    confidence = float(cls_confs[i]),
                    bbox_xyxy  = (x1, y1, x2, y2),
                ))

        return detections

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available – switching to CPU")
            return "cpu"
        return device


# ─── Pipeline ────────────────────────────────────────────────────────────────

class DetectionPipeline:
    """
    End-to-end inference + distance estimation + annotation pipeline.
    """

    def __init__(
        self,
        detector:  YOLODetector,
        estimator: DistanceEstimator,
        annotator: BBoxAnnotator,
    ):
        self.detector  = detector
        self.estimator = estimator
        self.annotator = annotator

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """
        Full forward pass on one frame.

        Returns:
            (annotated_frame, list_of_detections)
        """
        detections = self.detector.predict(frame)
        self.estimator.enrich_detections(detections)
        annotated  = self.annotator.draw(frame.copy(), detections)
        return annotated, detections


# ─── Source handlers ─────────────────────────────────────────────────────────

def run_on_image(pipeline: DetectionPipeline, img_path: str, save: bool, output_dir: str) -> None:
    """Detect on a single image file."""
    frame = cv2.imread(img_path)
    if frame is None:
        logger.error(f"Cannot read image: {img_path}")
        return

    annotated, dets = pipeline.process_frame(frame)

    # Print results
    logger.info(f"Detected {len(dets)} object(s) in {img_path}:")
    for d in dets:
        logger.info(f"  {d.label}  (conf={d.confidence:.2f}, bbox={d.bbox_xyxy})")

    # Display
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save
    if save:
        out_path = Path(output_dir) / ("det_" + Path(img_path).name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"Saved → {out_path}")


def run_on_video(
    pipeline: DetectionPipeline,
    source: str,
    save: bool,
    output_dir: str,
    display: bool = True,
) -> None:
    """Detect on a video file or webcam stream."""
    src    = int(source) if source.isdigit() else source
    cap    = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        out_path = Path(output_dir) / "output_detection.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (W, H))
        logger.info(f"Saving video → {out_path}")

    frame_count = 0
    t_total     = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        annotated, dets = pipeline.process_frame(frame)
        dt = time.perf_counter() - t0

        frame_count += 1
        t_total     += dt
        live_fps     = 1.0 / dt if dt > 0 else 0

        # FPS overlay
        cv2.putText(
            annotated, f"FPS: {live_fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8,
            (0, 255, 0), 2, cv2.LINE_AA,
        )

        if display:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_fps = frame_count / t_total if t_total > 0 else 0
    logger.info(f"Processed {frame_count} frames  |  avg FPS: {avg_fps:.1f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object detection + distance estimation")
    p.add_argument("--source",  required=True,
                   help="Image path, video path, or webcam index (0, 1, …)")
    p.add_argument("--weights", default="runs/train/robotics_nav_detection/weights/best.pt",
                   help="Path to YOLOv8 .pt or .onnx weights")
    p.add_argument("--config",  default="configs/config.yaml",
                   help="Project config YAML")
    p.add_argument("--conf",    type=float, default=None,
                   help="Confidence threshold override")
    p.add_argument("--device",  default=None,
                   help="Device override: cuda / cpu")
    p.add_argument("--save",    action="store_true",
                   help="Save annotated output")
    p.add_argument("--output",  default="outputs",
                   help="Output directory when --save is used")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress cv2.imshow (headless mode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg    = cfg["model"]
    dist_cfg     = cfg["distance"]

    conf_thres   = args.conf    or model_cfg["confidence_threshold"]
    device       = args.device  or cfg["training"]["device"]
    focal_length = dist_cfg["focal_length_px"]
    real_heights = dist_cfg["real_heights"]

    # ── Build pipeline ────────────────────────────────────────────────────────
    detector  = YOLODetector(
        weights    = args.weights,
        conf_thres = conf_thres,
        iou_thres  = model_cfg["iou_threshold"],
        device     = device,
        img_size   = cfg["dataset"]["image_size"],
    )
    estimator = DistanceEstimator(
        focal_length_px = focal_length,
        real_heights    = real_heights,
    )
    annotator = BBoxAnnotator()
    pipeline  = DetectionPipeline(detector, estimator, annotator)

    # ── Route to handler ──────────────────────────────────────────────────────
    src = args.source
    is_image = Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        run_on_image(pipeline, src, save=args.save, output_dir=args.output)
    else:
        run_on_video(
            pipeline, src,
            save=args.save,
            output_dir=args.output,
            display=not args.no_display,
        )


if __name__ == "__main__":
    main()
