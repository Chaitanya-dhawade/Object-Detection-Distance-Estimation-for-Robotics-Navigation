"""
utils/advanced_vision.py
─────────────────────────
Optional computer-vision algorithms for robotics navigation:

  1. Perspective Transform  → Bird's Eye View (BEV)
  2. Optical Flow Tracking  → Lucas-Kanade + dense Farnebäck
  3. Epipolar Geometry      → fundamental matrix, stereo epipolar lines

Each section contains:
  • Theoretical explanation (docstring)
  • Working OpenCV implementation
  • Demo function

Usage:
    python utils/advanced_vision.py --demo bev --source video.mp4
    python utils/advanced_vision.py --demo optical_flow --source 0
    python utils/advanced_vision.py --demo epipolar --left left.jpg --right right.jpg
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("advanced_vision")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PERSPECTIVE TRANSFORM – Bird's Eye View
# ══════════════════════════════════════════════════════════════════════════════

PERSPECTIVE_THEORY = """
Bird's Eye View (BEV) via Perspective Transform
──────────────────────────────────────────────
A perspective transform (homography) maps a planar region in one image to
another plane via a 3×3 matrix H:

    p' = H · p        (homogeneous coordinates)

For robotics:
  • We define four source points that mark a trapezoid on the road surface
    (visible in the normal camera view).
  • We specify four destination points that form a rectangle, representing
    what that same region should look like from directly above.
  • cv2.getPerspectiveTransform(src_pts, dst_pts) computes H.
  • cv2.warpPerspective(frame, H, output_size) applies it.

Why it matters:
  • Enables metric distance estimation on flat surfaces.
  • Simplifies lane / obstacle mapping.
  • Feeds downstream occupancy-grid planners.
"""


class BirdEyeView:
    """
    Perspective transform that produces a top-down (Bird's Eye View)
    image of the road ahead.

    Call `calibrate()` once (manually click 4 trapezoid corners or pass them),
    then call `transform()` on each frame.
    """

    def __init__(
        self,
        frame_size:  Tuple[int, int] = (1280, 720),
        output_size: Tuple[int, int] = (600, 400),
    ):
        self.frame_size  = frame_size   # (W, H) of input frame
        self.output_size = output_size  # (W, H) of BEV output
        self.H:  Optional[np.ndarray] = None   # homography matrix
        self.Hi: Optional[np.ndarray] = None   # inverse homography

    def calibrate(
        self,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
    ) -> None:
        """
        Compute homography from source → destination point pairs.

        If src_pts is None, uses default road trapezoid for a 1280×720 image.

        Args:
            src_pts: (4, 2) float32 – corners of road trapezoid (perspective view)
            dst_pts: (4, 2) float32 – corresponding BEV rectangle corners
        """
        W, H = self.frame_size
        bW, bH = self.output_size

        if src_pts is None:
            # Default road trapezoid (tune per camera / mount)
            src_pts = np.float32([
                [W * 0.43, H * 0.65],   # top-left
                [W * 0.57, H * 0.65],   # top-right
                [W * 0.90, H * 0.95],   # bottom-right
                [W * 0.10, H * 0.95],   # bottom-left
            ])

        if dst_pts is None:
            dst_pts = np.float32([
                [bW * 0.25, 0],
                [bW * 0.75, 0],
                [bW * 0.75, bH],
                [bW * 0.25, bH],
            ])

        self.H  = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.Hi = cv2.getPerspectiveTransform(dst_pts, src_pts)
        logger.info("BEV homography calibrated.")

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply perspective warp → Bird's Eye View."""
        if self.H is None:
            self.calibrate()
        return cv2.warpPerspective(frame, self.H, self.output_size)

    def inverse_transform(self, bev: np.ndarray) -> np.ndarray:
        """Warp BEV image back to perspective view."""
        if self.Hi is None:
            raise RuntimeError("Calibrate first.")
        return cv2.warpPerspective(bev, self.Hi, self.frame_size)

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the source trapezoid on the original frame."""
        if self.H is None:
            return frame
        # Recover src_pts from H – use a fixed default overlay instead
        W, H = self.frame_size
        pts = np.int32([
            [int(W*0.43), int(H*0.65)],
            [int(W*0.57), int(H*0.65)],
            [int(W*0.90), int(H*0.95)],
            [int(W*0.10), int(H*0.95)],
        ])
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        return frame


def demo_bev(source: str) -> None:
    """Live BEV demo on a video or webcam."""
    print(PERSPECTIVE_THEORY)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    bev = BirdEyeView()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]
        bev.frame_size = (W, H)
        bev.H = None   # recalibrate for actual frame size each time

        bev_frame = bev.transform(frame)
        combined  = np.hstack([
            cv2.resize(bev.draw_roi(frame.copy()), (640, 360)),
            cv2.resize(bev_frame, (640, 360)),
        ])
        cv2.putText(combined, "Original + ROI", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combined, "Bird's Eye View", (650, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("BEV Demo", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# 2. OPTICAL FLOW TRACKING
# ══════════════════════════════════════════════════════════════════════════════

OPTICAL_FLOW_THEORY = """
Optical Flow
────────────
Optical flow estimates the apparent motion of pixels between consecutive
frames, assuming brightness constancy:

    I(x, y, t) ≈ I(x+dx, y+dy, t+dt)

Two classic implementations are provided:

A) Lucas-Kanade (sparse):
   • Tracks a small set of good feature points (Shi-Tomasi corners).
   • Solves a local linear system per point.
   • Fast, accurate for rigid objects.
   • cv2.calcOpticalFlowPyrLK()

B) Farnebäck (dense):
   • Computes flow for every pixel using polynomial expansion.
   • Visualised as an HSV colour wheel (hue = direction, value = magnitude).
   • cv2.calcOpticalFlowFarneback()

Robotics use-cases:
  • Ego-motion estimation (how fast is the robot moving?).
  • Moving obstacle detection (flow vectors inconsistent with ego-motion).
  • Object tracking between detection frames.
"""


class LucasKanadeTracker:
    """
    Sparse optical flow tracker using Lucas-Kanade.
    Automatically re-detects features when too few remain.
    """

    LK_PARAMS = dict(
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    FEATURE_PARAMS = dict(
        maxCorners  = 100,
        qualityLevel= 0.3,
        minDistance = 7,
        blockSize   = 7,
    )

    def __init__(self, redetect_interval: int = 30):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts:  Optional[np.ndarray] = None
        self.frame_idx  = 0
        self.redetect_interval = redetect_interval
        self.track_colour = (0, 255, 0)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEATURE_PARAMS)
        return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame: track features, draw flow vectors.
        Returns annotated frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis  = frame.copy()

        if self.prev_gray is None or self.frame_idx % self.redetect_interval == 0:
            self.prev_pts = self._detect_features(gray)

        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.LK_PARAMS
            )
            good_new = next_pts[status.ravel() == 1]
            good_old = self.prev_pts[status.ravel() == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                cv2.arrowedLine(vis, (c, d), (a, b), self.track_colour, 2, tipLength=0.3)
                cv2.circle(vis, (a, b), 3, self.track_colour, -1)

            self.prev_pts = good_new.reshape(-1, 1, 2)
        else:
            self.prev_pts = self._detect_features(gray)

        self.prev_gray = gray
        self.frame_idx += 1
        return vis


class FarnebackFlow:
    """Dense optical flow visualiser using Farnebäck's algorithm."""

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Returns an HSV-encoded dense flow visualisation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2          # hue = direction
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.prev_gray = gray
        return bgr


def demo_optical_flow(source: str) -> None:
    """Side-by-side LK sparse and Farnebäck dense optical flow demo."""
    print(OPTICAL_FLOW_THEORY)
    cap  = cv2.VideoCapture(int(source) if source.isdigit() else source)
    lk   = LucasKanadeTracker()
    fb   = FarnebackFlow()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lk_vis = lk.process(frame.copy())
        fb_vis = fb.process(frame.copy())

        display = np.hstack([
            cv2.resize(lk_vis, (640, 360)),
            cv2.resize(fb_vis, (640, 360)),
        ])
        cv2.putText(display, "LK Sparse Flow", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Farneback Dense Flow", (650, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Optical Flow", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# 3. EPIPOLAR GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

EPIPOLAR_THEORY = """
Epipolar Geometry
─────────────────
Epipolar geometry describes the geometric relationship between two camera
views of the same 3D scene.

Key concepts:
  • Baseline: line connecting the two camera centres (O_L, O_R).
  • Epipole: projection of one camera centre onto the other image plane.
  • Epipolar plane: plane containing a 3D point P and both camera centres.
  • Epipolar line: intersection of the epipolar plane with an image plane.

Fundamental Matrix F  (3×3, rank 2):
    For corresponding points  p_L  and  p_R :
        p_R^T · F · p_L = 0
    F encodes only the rotation and translation between cameras
    (not intrinsic calibration).

Essential Matrix E  (for calibrated cameras):
    E = K_R^T · F · K_L
    Contains the actual rotation R and translation t:
        E = [t]× · R

Applications:
  • Stereo depth estimation (dense or sparse).
  • Structure-from-Motion (SfM) initialisation.
  • Outlier rejection (RANSAC on the epipolar constraint).

Implementation below:
  1. Detect SIFT keypoints in both images.
  2. Match with BFMatcher + Lowe's ratio test.
  3. Compute F via RANSAC (cv2.findFundamentalMat).
  4. Draw epipolar lines on both images.
"""


def compute_epipolar_lines(
    img_left:  np.ndarray,
    img_right: np.ndarray,
    num_points: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect correspondences, compute the Fundamental Matrix,
    and draw epipolar lines.

    Args:
        img_left, img_right:  BGR images from a stereo pair
        num_points:           Number of matches to visualise

    Returns:
        (img_left_vis, img_right_vis, F_matrix)
    """
    gray_l = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # ── Feature detection ─────────────────────────────────────────────────
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        logger.warning("Not enough features for epipolar computation")
        return img_left, img_right, np.eye(3)

    # ── Matching (Lowe ratio test) ────────────────────────────────────────
    bf      = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)
    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        logger.warning(f"Too few good matches ({len(good)}) for F computation")
        return img_left, img_right, np.eye(3)

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])

    # ── Fundamental matrix via RANSAC ─────────────────────────────────────
    F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, 3.0, 0.99)
    if F is None or F.shape != (3, 3):
        logger.warning("Fundamental matrix computation failed")
        return img_left, img_right, np.eye(3)

    inlier_mask = mask.ravel().astype(bool)
    pts_l = pts_l[inlier_mask]
    pts_r = pts_r[inlier_mask]

    # ── Draw epipolar lines ───────────────────────────────────────────────
    idxs   = np.random.choice(len(pts_l), min(num_points, len(pts_l)), replace=False)
    vis_l  = img_left.copy()
    vis_r  = img_right.copy()
    H, W   = img_left.shape[:2]

    lines_r = cv2.computeCorrespondEpilines(pts_l[idxs], 1, F).reshape(-1, 3)
    lines_l = cv2.computeCorrespondEpilines(pts_r[idxs], 2, F).reshape(-1, 3)

    for (a, b, c), pt_r, (a2, b2, c2), pt_l in zip(
        lines_r, pts_r[idxs], lines_l, pts_l[idxs]
    ):
        color = tuple(np.random.randint(50, 255, 3).tolist())
        # Epipolar line on right image
        x0, y0 = 0, int(-c / b) if b != 0 else 0
        x1, y1 = W, int(-(a * W + c) / b) if b != 0 else H
        cv2.line(vis_r, (x0, y0), (x1, y1), color, 1)
        cv2.circle(vis_r, tuple(pt_r.astype(int)), 5, color, -1)
        # Epipolar line on left image
        x0, y0 = 0, int(-c2 / b2) if b2 != 0 else 0
        x1, y1 = W, int(-(a2 * W + c2) / b2) if b2 != 0 else H
        cv2.line(vis_l, (x0, y0), (x1, y1), color, 1)
        cv2.circle(vis_l, tuple(pt_l.astype(int)), 5, color, -1)

    inlier_count = int(inlier_mask.sum())
    cv2.putText(vis_l, f"F-inliers: {inlier_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return vis_l, vis_r, F


def demo_epipolar(left_path: str, right_path: str) -> None:
    """Visualise epipolar lines from a stereo image pair."""
    print(EPIPOLAR_THEORY)
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    if img_l is None or img_r is None:
        logger.error("Cannot read stereo images")
        return

    vis_l, vis_r, F = compute_epipolar_lines(img_l, img_r)
    display = np.hstack([
        cv2.resize(vis_l, (640, 360)),
        cv2.resize(vis_r, (640, 360)),
    ])
    cv2.putText(display, "Left + epipolar lines", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, "Right + epipolar lines", (650, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    print(f"\nFundamental Matrix F:\n{F}\n")
    cv2.imshow("Epipolar Geometry", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced vision demos")
    p.add_argument("--demo", choices=["bev", "optical_flow", "epipolar"],
                   default="bev")
    p.add_argument("--source", default="0",
                   help="Video path or webcam index (for bev / optical_flow)")
    p.add_argument("--left",  default=None, help="Left image (epipolar)")
    p.add_argument("--right", default=None, help="Right image (epipolar)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.demo == "bev":
        demo_bev(args.source)
    elif args.demo == "optical_flow":
        demo_optical_flow(args.source)
    elif args.demo == "epipolar":
        if not args.left or not args.right:
            print("Provide --left and --right image paths for epipolar demo")
        else:
            demo_epipolar(args.left, args.right)
