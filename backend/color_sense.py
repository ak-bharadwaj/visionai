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


def expand_bbox(x1: int, y1: int, x2: int, y2: int,
                frame_w: int, frame_h: int,
                scale: float = 0.30,
                class_name: str = "") -> tuple:
    """
    Expand a bounding box outward by `scale` fraction of the box dimensions.

    For persons: biases the expansion downward to capture the torso/shirt
    region rather than the face.  YOLO person boxes tightly fit the full
    body but the centroid is often near the head — clothing is in the lower
    half.  Shifting the crop window downward by 15 % of the box height and
    expanding the bottom by an additional 35 % isolates the shirt region.

    For all other classes: uniform 30 % expansion in every direction.

    All coordinates are clamped to the frame boundary.
    """
    bw = x2 - x1
    bh = y2 - y1

    if class_name.lower() == "person":
        # Torso-biased expansion: push top down, pull bottom down further.
        # The result is a crop centred on the chest/shirt area.
        new_x1 = max(0,       int(x1 - bw * scale))
        new_y1 = max(0,       int(y1 + bh * 0.15))   # shift top DOWN 15 %
        new_x2 = min(frame_w, int(x2 + bw * scale))
        new_y2 = min(frame_h, int(y2 + bh * 0.35))   # extend bottom DOWN 35 %
    else:
        new_x1 = max(0,       int(x1 - bw * scale))
        new_y1 = max(0,       int(y1 - bh * scale))
        new_x2 = min(frame_w, int(x2 + bw * scale))
        new_y2 = min(frame_h, int(y2 + bh * scale))

    # Sanity check — if expansion collapsed the box (e.g. person at very top),
    # fall back to the original clamped bbox.
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return (
            max(0, x1), max(0, y1),
            min(frame_w, x2), min(frame_h, y2),
        )
    return new_x1, new_y1, new_x2, new_y2


def _get_person_shirt_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                             frame_w: int, frame_h: int) -> np.ndarray:
    """
    Extract ONLY the shirt/torso region - EXCLUDE face (skin tone = orange/warm).
    Person bbox: top = head, bottom = torso. Use lower 55% only (skip top 45% = face/neck).
    """
    bh = y2 - y1
    bw = x2 - x1
    if bh < 30 or bw < 30:  # Too small - use center crop
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        h, w = frame.shape[:2]
        return frame[max(0,cy-40):min(h,cy+40), max(0,cx-40):min(w,cx+40)]
    # Start at 40% down - skip face entirely, get chest/shirt
    shirt_y1 = y1 + int(bh * 0.40)
    shirt_y2 = y2
    # Use center 70% horizontally - avoid arms/background at edges
    shirt_x1 = x1 + int(bw * 0.15)
    shirt_x2 = x2 - int(bw * 0.15)
    shirt_y1 = max(0, min(shirt_y1, frame_h - 10))
    shirt_y2 = max(shirt_y1 + 20, min(shirt_y2, frame_h))
    shirt_x1 = max(0, min(shirt_x1, frame_w - 10))
    shirt_x2 = max(shirt_x1 + 20, min(shirt_x2, frame_w))
    region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if region.size < 100:  # Fallback if region too small
        return frame[y1:y2, x1:x2]  # Use full bbox as last resort
    return region


def get_dominant_color(frame: np.ndarray, det=None) -> str:
    """
    Google-level: Improved color detection - EXCLUDES face/skin for person (avoids orange).
    For person: samples ONLY shirt/torso region. For others: center of object.
    """
    try:
        if det is not None and hasattr(det, 'x1'):
            frame_h, frame_w = frame.shape[:2]
            class_name = getattr(det, "class_name", "")
            
            # CRITICAL: For person, use ONLY shirt region - NEVER include face (skin = orange)
            if class_name.lower() == "person":
                region = _get_person_shirt_region(
                    frame, det.x1, det.y1, det.x2, det.y2, frame_w, frame_h
                )
            else:
                # Other objects: use center 60% of bbox
                ex1, ey1, ex2, ey2 = expand_bbox(
                    det.x1, det.y1, det.x2, det.y2,
                    frame_w, frame_h, scale=0.15, class_name=class_name,
                )
                bbox_w = ex2 - ex1
                bbox_h = ey2 - ey1
                center_x = (ex1 + ex2) // 2
                center_y = (ey1 + ey2) // 2
                crop_w = int(bbox_w * 0.6)
                crop_h = int(bbox_h * 0.6)
                crop_x1 = max(ex1, center_x - crop_w // 2)
                crop_y1 = max(ey1, center_y - crop_h // 2)
                crop_x2 = min(ex2, crop_x1 + crop_w)
                crop_y2 = min(ey2, crop_y1 + crop_h)
                region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            # Tight center crop — 20% of frame width/height.
            h, w = frame.shape[:2]
            cy0, cy1 = int(h * 0.40), int(h * 0.60)
            cx0, cx1 = int(w * 0.40), int(w * 0.60)
            region = frame[cy0:cy1, cx0:cx1]

        if region.size == 0:
            return "unknown"

        # Google-level: Use both HSV and RGB for better accuracy
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Calculate statistics
        mean_s = float(np.mean(hsv[:, :, 1]))  # Saturation
        mean_v = float(np.mean(hsv[:, :, 2]))  # Value (brightness)
        mean_h = float(np.median(hsv[:, :, 0]))  # Hue (median for robustness)
        
        # Google-level: Better handling of desaturated colors (grey, white, black)
        # Low saturation = grey/white/black - NOT orange (that was skin tone)
        if mean_s < 35:  # Very low saturation = neutral colors
            if mean_v < 45:
                return "black"
            elif mean_v < 90:
                return "dark gray"
            elif mean_v > 200:
                return "white"
            else:
                return "gray"
        
        # Google-level: Use RGB to verify color (more reliable for light colors)
        mean_r = float(np.mean(rgb[:, :, 0]))
        mean_g = float(np.mean(rgb[:, :, 1]))
        mean_b = float(np.mean(rgb[:, :, 2]))
        
        # Google-level: Detect light blue/grey (common issue)
        # Light blue has: B > R and B > G, but low saturation
        if mean_s < 60 and mean_b > mean_r and mean_b > mean_g:
            if mean_v > 150:
                return "light blue"
            else:
                return "blue"
        
        # Google-level: Detect light grey/blue (desaturated blue)
        if mean_s < 50:
            # Check if it's a desaturated version of a color
            rgb_max = max(mean_r, mean_g, mean_b)
            rgb_min = min(mean_r, mean_g, mean_b)
            if rgb_max - rgb_min < 30:  # Very similar RGB = grey
                return "gray"
            elif mean_b > mean_r and mean_b > mean_g:
                return "light blue"
        
        # Google-level: Use hue for saturated colors
        base_color = _hue_to_name(mean_h)
        
        # Google-level: Better brightness classification
        if mean_v < 70:
            prefix = "dark "
        elif mean_v > 200:
            prefix = "light "
        else:
            prefix = ""
        
        return f"{prefix}{base_color}"
    except Exception as e:
        logger.error(f"Color sense error: {e}")
        return "unknown"

def answer_color_question(frame: np.ndarray, detections: list) -> str:
    """
    Google-level: Improved color question handler.
    Better object selection and more accurate color detection.
    """
    det = None
    if detections:
        frame_cx = frame.shape[1] / 2
        frame_cy = frame.shape[0] / 2
        # Google-level: Better scoring - prefer person/object in center
        def score(d):
            area = (d.x2 - d.x1) * (d.y2 - d.y1) if hasattr(d, 'x2') else 0
            if area == 0:
                return 0
            dx   = ((d.x1 + d.x2) / 2 - frame_cx) if hasattr(d, 'x1') else 999
            dy   = ((d.y1 + d.y2) / 2 - frame_cy) if hasattr(d, 'y1') else 999
            dist = (dx**2 + dy**2) ** 0.5
            # Google-level: Strong preference for center objects
            center_score = 1.0 / (1.0 + dist * 0.02)  # Stronger center preference
            # Google-level: Prefer "person" for clothing questions
            class_name = getattr(d, 'class_name', '')
            class_boost = 2.0 if class_name == 'person' else 1.0
            return area * center_score * class_boost
        det = max(detections, key=score)
    
    # Google-level: Get color with improved algorithm
    color = get_dominant_color(frame, det)
    
    # Google-level: Better response formatting
    if det and hasattr(det, 'class_name'):
        class_name = det.class_name
        # Google-level: Handle clothing-specific responses
        if class_name == 'person':
            return f"The person's clothing appears to be {color}."
        else:
            return f"The {class_name} appears to be {color}."
    return f"The object in front appears to be {color}."
