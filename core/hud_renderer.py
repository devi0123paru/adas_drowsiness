"""
core/hud_renderer.py
--------------------
Draws the HUD overlay onto the live camera frame.
"""

import cv2
import numpy as np
import time
from core.perception import PerceptionFrame, TemporalFeatures
from core.fusion import FusionOutput, AlertLevel
from core.glare_detection import GlareDetectionFrame

C = {
    "ok":        (0,   210,  90),
    "advisory":  (100, 240, 120),
    "warning":   (0,   165, 255),
    "critical":  (0,   100, 255),
    "emergency": (0,    40, 230),
    "muted":     (80,   90, 110),
    "text":      (200, 210, 230),
    "accent":    (0,   220, 255),
}

ALERT_COL = {
    AlertLevel.NONE:      C["ok"],
    AlertLevel.ADVISORY:  C["advisory"],
    AlertLevel.WARNING:   C["warning"],
    AlertLevel.CRITICAL:  C["critical"],
    AlertLevel.EMERGENCY: C["emergency"],
}

ALERT_BG = {
    AlertLevel.NONE:      (10,  14,  18),
    AlertLevel.ADVISORY:  (10,  30,  10),
    AlertLevel.WARNING:   (10,  40,  60),
    AlertLevel.CRITICAL:  (10,  20, 100),
    AlertLevel.EMERGENCY: (10,   8, 160),
}

ALERT_LABEL = {
    AlertLevel.NONE:      "● NOMINAL",
    AlertLevel.ADVISORY:  "◈ ADVISORY",
    AlertLevel.WARNING:   "⚠ WARNING",
    AlertLevel.CRITICAL:  "✖ CRITICAL",
    AlertLevel.EMERGENCY: "EMERGENCY",
}

def _put(img, text, pos, scale=0.5, color=C["text"], thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def _bar(img, x, y, w, h, value, max_val, color, bg=(30, 35, 45)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * min(value / max(max_val, 1e-6), 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)

def _reduce_glare(img, gdf: GlareDetectionFrame) -> np.ndarray:
    """
    Reduce glare by dimming only the brightest regions (headlights).
    Preserves the rest of the scene for better driver visibility.
    
    Algorithm:
    1. Convert to HSV to get brightness (Value channel)
    2. Create mask of bright regions above warning threshold
    3. Reduce brightness in masked areas proportional to glare severity
    4. Blend back to preserve detail
    """
    if not gdf.is_glare_detected or gdf.glare_severity <= 0:
        return img
    
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v_channel = hsv[:, :, 2]
        
        # Create mask of bright regions (above warning threshold)
        # Typically warning is 120 lux, so mask pixels with high brightness
        bright_threshold = 180  # Target bright headlight regions
        mask = (v_channel > bright_threshold).astype(np.float32)
        
        # Expand mask slightly for smooth transition
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / 255.0  # Normalize to 0-1
        
        # Reduction strength based on glare severity
        # At 50% severity: reduce brightness by 30%
        # At 100% severity: reduce brightness by 60%
        reduction = 0.3 + (gdf.glare_severity / 100.0) * 0.3
        
        # Apply reduction only to bright regions
        v_reduced = v_channel.copy()
        v_reduced = v_reduced * (1.0 - mask * reduction)
        
        # Clamp to valid range
        v_reduced = np.clip(v_reduced, 0, 255)
        
        # Put reduced values back
        hsv[:, :, 2] = v_reduced
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    except Exception:
        return img


class HUDRenderer:

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def render(self, frame: np.ndarray,
               pf: PerceptionFrame,
               tf: TemporalFeatures,
               fo: FusionOutput,
               gdf: GlareDetectionFrame,
               fps: float,
               calib_progress: float) -> np.ndarray:

        h, w = frame.shape[:2]
        al   = fo.alert_level
        col  = ALERT_COL[al]

        # ── SMART GLARE REDUCTION: Dim only bright regions (headlights) ──────
        frame = _reduce_glare(frame, gdf)

        # ── Top bar ──────────────────────────────────────────────────────────
        bar_h = 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), ALERT_BG[al], -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), col, 1)

        # Status label
        _put(frame, ALERT_LABEL[al], (16, 46),
             scale=1.15, color=col, thickness=3)

        # Calibration bar
        if not fo.is_calibrated:
            bw = 230
            _bar(frame, 16, 60, bw, 12, calib_progress, 1.0, (0, 200, 120))
            _put(frame, f"CALIBRATING {int(calib_progress * 100)}%",
                 (16, 58), 0.38, C["accent"])
        else:
            _put(frame, f"CAN {fo.can_frame}", (16, 80), 0.34, C["muted"])

        # Glare detection indicator
        glare_col = (0, 0, 200) if gdf.glare_severity < 25 else (0, 165, 255) if gdf.glare_severity < 50 else (0, 40, 230)
        glare_label = "⚡ HEADLIGHTS!" if gdf.glare_severity > 50 else "⚠ BRIGHT LIGHT" if gdf.is_glare_detected else "○ CLEAR"
        glare_detail = f"{gdf.light_category}"
        _put(frame, glare_label, (300, 46), 0.45, glare_col, 1)
        _put(frame, glare_detail, (300, 62), 0.32, (100, 100, 100), 1)
        _bar(frame, 300, 70, 80, 8, gdf.light_intensity, 255.0, glare_col)
        _put(frame, f"{int(gdf.light_intensity)}L", (390, 76), 0.32, C["muted"])

        # Attention arc gauge
        score = fo.attention_score
        gc    = (0, int(score * 2), int(255 - score * 2))
        cx, cy, r = w - 55, 46, 32
        cv2.ellipse(frame, (cx, cy), (r, r), -135, 0, 270, (40, 45, 55), 5)
        arc_angle = int(270 * score / 100)
        if arc_angle > 0:
            cv2.ellipse(frame, (cx, cy), (r, r), -135, 0, arc_angle, gc, 4)
        _put(frame, f"{int(score)}", (cx - 14, cy + 7), 0.72, gc, 2)
        _put(frame, "ATTN", (cx - 15, cy + 22), 0.35, C["muted"])

        # KSS and fatigue
        _put(frame, f"KSS {fo.kss_estimate:.1f}/9",
             (w - 160, 28), 0.46, C["text"])
        _put(frame, f"P(fat) {fo.fatigue_probability:.2f}",
             (w - 160, 48), 0.42, C["muted"])
        _put(frame, f"{fps:.0f}fps", (w - 160, 82), 0.38, C["muted"])

        # ── Right metrics panel ───────────────────────────────────────────────
        px = w - 190
        ov2 = frame.copy()
        cv2.rectangle(ov2, (px - 8, bar_h), (w, h), (10, 13, 20), -1)
        cv2.addWeighted(ov2, 0.62, frame, 0.38, 0, frame)
        cv2.line(frame, (px - 8, bar_h), (px - 8, h), (30, 35, 50), 1)

        def metric(label, val_str, y_off, warn=False, danger=False):
            mc = C["emergency"] if danger else (C["warning"] if warn else C["text"])
            yy = bar_h + 22 + y_off
            _put(frame, label,   (px, yy),      0.37, C["muted"])
            _put(frame, val_str, (px + 88, yy), 0.46, mc, 1)

        metric("EAR",
               f"{pf.ear_smooth:.3f}", 0,
               pf.ear_smooth < 0.22, pf.ear_smooth < 0.18)
        metric("MAR",
               f"{pf.mar:.3f}", 22,
               pf.mar > 0.55, pf.mar > 0.70)
        metric("PERCLOS",
               f"{tf.perclos * 100:.1f}%", 44,
               tf.perclos > 0.15, tf.perclos > 0.30)
        metric("BLINK/MIN",
               f"{tf.blink_rate:.1f}", 66,
               tf.blink_rate > 25 or tf.blink_rate < 8)
        metric("BLINK DUR",
               f"{tf.blink_duration:.0f}ms", 88,
               tf.blink_duration > 350)
        metric("PITCH",
               f"{pf.pitch_smooth:.1f}°", 110,
               abs(pf.pitch_smooth) > 15, abs(pf.pitch_smooth) > 25)
        metric("YAW",
               f"{pf.yaw_smooth:.1f}°", 132,
               abs(pf.yaw_smooth) > 20, abs(pf.yaw_smooth) > 40)
        metric("GAZE X",
               f"{pf.gaze_x:.2f}", 154,
               abs(pf.gaze_x) > 0.25)
        metric("YAWNS",
               str(tf.yawn_count_recent), 176,
               tf.yawn_count_recent >= 2)
        metric("uSLEEPS",
               str(tf.microsleep_count), 198,
               tf.microsleep_count > 0, tf.microsleep_count > 2)

        # Sub-score bars
        by0 = bar_h + 230
        for i, (lbl, val) in enumerate([
            ("EAR",     fo.score_ear),
            ("PERCLOS", fo.score_perclos),
            ("BLINK",   fo.score_blink),
            ("POSE",    fo.score_pose),
            ("YAWN",    fo.score_yawn),
            ("GLARE",   fo.score_glare),
        ]):
            by  = by0 + i * 20
            bc  = (0, int(val * 2), int(255 - val * 2))
            _put(frame, lbl, (px, by + 9), 0.34, C["muted"])
            _bar(frame, px + 52, by, 115, 7, val, 100, bc)
            _put(frame, f"{int(val)}", (px + 172, by + 9), 0.34, bc)

        # ── Head pose compass ─────────────────────────────────────────────────
        comp_cx = w // 2
        comp_cy = h - 45
        comp_r  = 32
        cv2.circle(frame, (comp_cx, comp_cy), comp_r, (22, 28, 40), -1)
        cv2.circle(frame, (comp_cx, comp_cy), comp_r, (45, 52, 70), 1)
        cv2.line(frame,
                 (comp_cx, comp_cy - comp_r + 4),
                 (comp_cx, comp_cy + comp_r - 4), (40, 46, 60), 1)
        cv2.line(frame,
                 (comp_cx - comp_r + 4, comp_cy),
                 (comp_cx + comp_r - 4, comp_cy), (40, 46, 60), 1)
        dot_x = comp_cx + int(
            np.clip(pf.yaw_smooth / 45, -1, 1) * comp_r * 0.8)
        dot_y = comp_cy + int(
            np.clip(pf.pitch_smooth / 45, -1, 1) * comp_r * 0.8)
        cv2.circle(frame, (dot_x, dot_y), 5, col, -1)
        _put(frame, "POSE", (comp_cx - 14, h - 10), 0.34, C["muted"])

        # ── No face warning ───────────────────────────────────────────────────
        if not pf.face_detected:
            _put(frame,
                 f"FACE LOST  {fo.consecutive_no_face_s:.1f}s",
                 (16, bar_h + 40), 0.72, C["critical"], 2)

        # ── Danger flash border ───────────────────────────────────────────────
        if al >= AlertLevel.CRITICAL:
            if int(time.time() * 4) % 2:
                thickness = 12 if al == AlertLevel.EMERGENCY else 8
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                               col, thickness)

        return frame