#!/usr/bin/env python3
"""
tools/replay_dataset.py — Offline dataset replay tool for VisionTalk pipeline.

Reads frames from a video file and drives them through the core detection,
tracking, depth, and risk-scoring pipeline — with no TTS, no WebSocket, and
no dependency on pipeline.py or main.py.

Outputs:
  - Per-frame CSV to stdout (or --output <file>):
      frame_idx, n_detections, n_confirmed, best_risk_level, best_dist_m, best_class
  - Summary statistics printed to stderr after all frames.

Usage:
  python tools/replay_dataset.py <video_path> [--output results.csv]
  python tools/replay_dataset.py <video_path> --max-frames 500
  python tools/replay_dataset.py <video_path> --conf 0.35

Dependencies: opencv-python, numpy — already required by the main pipeline.
"""

import argparse
import csv
import sys
import os
import time

# Allow imports from the project root regardless of where this script is run from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from backend.detector import Detector, Detection
from backend.tracker import ObjectTracker
from backend.depth import DepthEstimator
from backend.risk_engine import score_all
from backend.diagnostics import Diagnostics


# ── Constants ────────────────────────────────────────────────────────────────

RISK_LEVEL_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0, "": 0}

CSV_FIELDS = [
    "frame_idx",
    "n_detections",
    "n_confirmed",
    "best_risk_level",
    "best_risk_score",
    "best_dist_m",
    "best_class",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_object(confirmed):
    """Return the highest-risk confirmed object, or None."""
    if not confirmed:
        return None
    return max(confirmed, key=lambda o: (
        RISK_LEVEL_ORDER.get(o.risk_level, 0), o.risk_score
    ))


def _parse_args():
    p = argparse.ArgumentParser(description="VisionTalk offline pipeline replay")
    p.add_argument("video", help="Path to the input video file")
    p.add_argument(
        "--output", "-o",
        default=None,
        help="CSV output file path (default: stdout)",
    )
    p.add_argument(
        "--max-frames", "-n",
        type=int, default=None,
        help="Stop after this many frames (default: all frames)",
    )
    p.add_argument(
        "--conf",
        type=float, default=None,
        help="Override detection confidence threshold (default: CONF_DETECT from tracker.py)",
    )
    p.add_argument(
        "--no-depth",
        action="store_true",
        help="Skip MiDaS depth estimation (faster, distances will be 0.0)",
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Validate input
    if not os.path.isfile(args.video):
        print(f"ERROR: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[replay] Video: {args.video}  "
        f"frames={total_frames_in_file}  fps={fps_video:.1f}  "
        f"resolution={frame_w}x{frame_h}",
        file=sys.stderr,
    )

    # Initialise pipeline components
    detector = Detector(conf_threshold=args.conf)  # uses CONF_DETECT if None
    tracker = ObjectTracker()
    depth_estimator = DepthEstimator() if not args.no_depth else None
    diag = Diagnostics()

    # CSV output
    out_file = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    # Stats accumulators
    total_confirmed_sum = 0
    total_detections_sum = 0
    frames_processed = 0
    t_wall_start = time.monotonic()

    try:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            # ── Detection ────────────────────────────────────────────────────
            detections: list[Detection] = detector.detect(bgr)
            n_det = len(detections)
            total_detections_sum += n_det

            # ── Tracking ─────────────────────────────────────────────────────
            confirmed = tracker.update(detections, frame_w, frame_h)
            n_conf = len(confirmed)
            total_confirmed_sum += n_conf

            # ── Depth (optional) ─────────────────────────────────────────────
            if depth_estimator is not None and confirmed:
                # Run MiDaS on the frame
                depth_map = depth_estimator.estimate(bgr)
                for obj in confirmed:
                    depth_score = depth_estimator.get_region_depth(
                        depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                    )
                    if depth_score > 0.0:
                        dist_m = depth_estimator.metres_from_score(depth_score)
                        if not depth_estimator.depth_jump_reject(
                            obj.smoothed_distance_m, dist_m
                        ):
                            obj.update_distance(dist_m)
                            diag.depth_measurement_accepted()
                        else:
                            obj._depth_stale_frames += 1
                            diag.depth_measurement_rejected("jump_reject")

            # ── Risk scoring ─────────────────────────────────────────────────
            score_all(confirmed, frame_w=frame_w, frame_h=frame_h)

            # ── Best object for this frame ────────────────────────────────────
            best = _best_object(confirmed)
            row = {
                "frame_idx":      frame_idx,
                "n_detections":   n_det,
                "n_confirmed":    n_conf,
                "best_risk_level": best.risk_level if best else "",
                "best_risk_score": f"{best.risk_score:.4f}" if best else "0.0000",
                "best_dist_m":    f"{best.smoothed_distance_m:.3f}" if best else "0.000",
                "best_class":     best.class_name if best else "",
            }
            writer.writerow(row)

            frames_processed += 1
            frame_idx += 1

            # Progress feedback every 100 frames
            if frame_idx % 100 == 0:
                elapsed = time.monotonic() - t_wall_start
                print(
                    f"[replay] frame {frame_idx}/{total_frames_in_file}  "
                    f"elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )

    finally:
        cap.release()
        if args.output:
            out_file.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    t_elapsed = time.monotonic() - t_wall_start
    mean_confirmed = total_confirmed_sum / max(frames_processed, 1)
    mean_detections = total_detections_sum / max(frames_processed, 1)
    diag_sum = diag.summary()

    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("REPLAY SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Video              : {args.video}", file=sys.stderr)
    print(f"  Frames processed   : {frames_processed}", file=sys.stderr)
    print(f"  Wall time          : {t_elapsed:.2f} s", file=sys.stderr)
    print(f"  Effective FPS      : {frames_processed / max(t_elapsed, 0.001):.1f}", file=sys.stderr)
    print(f"  Mean detections/frame : {mean_detections:.2f}", file=sys.stderr)
    print(f"  Mean confirmed/frame  : {mean_confirmed:.2f}", file=sys.stderr)
    print(f"  Tracks created     : {diag_sum['tracks_created']}", file=sys.stderr)
    print(f"  Tracks revived     : {diag_sum['tracks_revived']}", file=sys.stderr)
    print(f"  Revival rate       : {diag_sum['revival_rate']:.1%}", file=sys.stderr)
    print(f"  Fragmentation rate : {diag_sum['track_fragmentation_rate']:.1%}", file=sys.stderr)
    if not args.no_depth:
        print(f"  Depth measurements : {diag_sum['depth_measurements']}", file=sys.stderr)
        print(f"  Depth failure rate : {diag_sum['depth_failure_rate']:.1%}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
