import cv2, numpy as np, logging

logger = logging.getLogger(__name__)

# OpenCV HSV: Hue is 0-179 (half of standard 0-360)
COLOR_RANGES = [
    (  0,  10, "red"),
    ( 10,  25, "orange"),
    ( 25,  35, "yellow"),
    ( 35,  85, "green"),
    ( 85, 100, "cyan"),
    (100, 130, "blue"),
    (130, 145, "purple"),
    (145, 160, "pink"),
    (160, 179, "red"),
]

def _hue_to_name(hue: float) -> str:
    for lo, hi, name in COLOR_RANGES:
        if lo <= hue <= hi:
            return name
    return "red"

def get_dominant_color(frame: np.ndarray, det=None) -> str:
    """
    Uses HSV colorspace to find dominant color.
    det: any object with x1,y1,x2,y2 (Detection or SpatialResult).
    If det=None: uses tight center 20% crop — user should hold object in frame center.
    Returns: e.g. "dark blue", "light green", "gray", "black", "white"
    """
    try:
        if det is not None and hasattr(det, 'x1'):
            region = frame[det.y1:det.y2, det.x1:det.x2]
        else:
            # Tight center crop — 20% of frame width/height.
            # If user holds a pen in the center, this reads the pen, not background.
            h, w = frame.shape[:2]
            cy0, cy1 = int(h * 0.40), int(h * 0.60)   # 20% height band
            cx0, cx1 = int(w * 0.40), int(w * 0.60)   # 20% width band
            region = frame[cy0:cy1, cx0:cx1]

        if region.size == 0:
            return "unknown"

        hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        if mean_s < 40:
            if mean_v < 60:    return "black"
            elif mean_v > 180: return "white"
            else:              return "gray"

        mean_h     = float(np.median(hsv[:, :, 0]))
        base_color = _hue_to_name(mean_h)

        if mean_v < 80:    prefix = "dark "
        elif mean_v > 200: prefix = "light "
        else:              prefix = ""

        return f"{prefix}{base_color}"
    except Exception as e:
        logger.error(f"Color sense error: {e}")
        return "unknown"

def answer_color_question(frame: np.ndarray, detections: list) -> str:
    """
    Color question handler.
    Priority: largest object by area closest to frame center → most likely what user is holding.
    Fallback: tight 20% center crop (works for pen, cup, any held object).
    """
    det = None
    if detections:
        frame_cx = frame.shape[1] / 2
        frame_cy = frame.shape[0] / 2
        # Score = area / (1 + distance_from_center) — prefer large AND central objects
        def score(d):
            area = (d.x2 - d.x1) * (d.y2 - d.y1) if hasattr(d, 'x2') else 0
            dx   = ((d.x1 + d.x2) / 2 - frame_cx) if hasattr(d, 'x1') else 999
            dy   = ((d.y1 + d.y2) / 2 - frame_cy) if hasattr(d, 'y1') else 999
            dist = (dx**2 + dy**2) ** 0.5
            return area / (1 + dist * 0.01)   # slight distance penalty
        det = max(detections, key=score)
    color = get_dominant_color(frame, det)
    if det and hasattr(det, 'class_name'):
        return f"The {det.class_name} appears to be {color}."
    return f"The object in front appears to be {color}."
