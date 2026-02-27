"""
scanner.py — QR code and barcode detection using OpenCV built-ins.
No external dependencies beyond opencv-python (already required by YOLO).

Detects:
  - QR codes   → cv2.QRCodeDetector
  - 1D/2D barcodes → cv2.barcode_BarcodeDetector (EAN-13, UPC-A, Code-128, etc.)
"""
import cv2
import logging

logger = logging.getLogger(__name__)


class Scanner:
    def __init__(self):
        self._qr  = cv2.QRCodeDetector()
        try:
            self._bc = cv2.barcode_BarcodeDetector()
        except Exception:
            self._bc = None
            logger.warning("[Scanner] barcode_BarcodeDetector unavailable — QR-only mode.")

    def scan(self, frame) -> list[dict]:
        """
        Scan frame for QR codes and barcodes.
        Returns list of dicts: [{type, value, bbox}]
        """
        results = []

        # ── QR codes ────────────────────────────────────────────────
        try:
            val, pts, _ = self._qr.detectAndDecode(frame)
            if val:
                bbox = _pts_to_bbox(pts)
                results.append({"type": "QR", "value": val, "bbox": bbox})
        except Exception as e:
            logger.debug(f"[Scanner] QR error: {e}")

        # ── Barcodes ────────────────────────────────────────────────
        if self._bc is not None:
            try:
                ok, decoded_info, decoded_type, pts = self._bc.detectAndDecodeMulti(frame)
                if ok and decoded_info:
                    for val, btype, bpts in zip(decoded_info, decoded_type, pts):
                        if val:
                            bbox = _pts_to_bbox(bpts)
                            results.append({
                                "type": btype if isinstance(btype, str) else "Barcode",
                                "value": val,
                                "bbox": bbox,
                            })
            except Exception as e:
                logger.debug(f"[Scanner] Barcode error: {e}")

        return results

    def format_result(self, results: list[dict]) -> str:
        """Convert scan results to a human-readable speech string."""
        if not results:
            return ""
        parts = []
        for r in results:
            t   = r.get("type", "Code")
            val = r.get("value", "").strip()
            if not val:
                continue
            # Classify value for better speech
            if t == "QR" and (val.startswith("http://") or val.startswith("https://")):
                parts.append(f"QR code links to: {val}")
            elif t == "QR":
                parts.append(f"QR code reads: {val}")
            elif t in ("EAN_13", "EAN_8", "UPC_A", "UPC_E"):
                parts.append(f"Barcode {t}: {val}")
            else:
                parts.append(f"{t} code: {val}")
        return ". ".join(parts)


def _pts_to_bbox(pts) -> dict | None:
    """Convert detector point array to {x1, y1, x2, y2} bbox."""
    if pts is None:
        return None
    try:
        import numpy as np
        pts = np.array(pts).reshape(-1, 2)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except Exception:
        return None


scanner = Scanner()
