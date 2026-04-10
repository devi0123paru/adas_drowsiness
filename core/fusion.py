"""
core/fusion.py
--------------
Combines all signals into one attention score and runs the alert state machine.
Integrates drowsiness signals (facial) with environmental factors (glare detection).
"""

import time
import math
from dataclasses import dataclass
from typing import Optional
from core.perception import PerceptionFrame, TemporalFeatures
from core.glare_detection import GlareDetectionFrame

class AlertLevel:
    NONE      = 0
    ADVISORY  = 1
    WARNING   = 2
    CRITICAL  = 3
    EMERGENCY = 4

ALERT_LABELS = {
    0: "NOMINAL",
    1: "ADVISORY",
    2: "WARNING",
    3: "CRITICAL",
    4: "EMERGENCY",
}

@dataclass
class FusionOutput:
    timestamp: float = 0.0
    attention_score: float     = 100.0
    kss_estimate: float        = 1.0
    fatigue_probability: float = 0.0
    alert_level: int  = 0
    alert_label: str  = "NOMINAL"
    alert_new: bool   = False
    score_ear:    float = 100.0
    score_perclos:float = 100.0
    score_blink:  float = 100.0
    score_pose:   float = 100.0
    score_yawn:   float = 100.0
    score_glare:  float = 100.0
    glare_severity: float = 0.0
    is_calibrated: bool = False
    face_present:  bool = False
    consecutive_no_face_s: float = 0.0
    can_frame: str = ""

def attention_to_kss(score: float) -> float:
    return 1.0 + (1.0 - score / 100.0) * 8.0

def fatigue_probability(score: float) -> float:
    x = (50.0 - score) / 15.0
    return 1.0 / (1.0 + math.exp(-x))

class FusionEngine:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fc = cfg["fusion"]
        ac = cfg["alerts"]
        pc = cfg["perception"]

        self.w_ear     = fc["weight_ear"]
        self.w_perclos = fc["weight_perclos"]
        self.w_blink   = fc["weight_blink_rate"]
        self.w_pitch   = fc["weight_head_pitch"]
        self.w_yaw     = fc["weight_head_yaw"]
        self.w_yawn    = fc["weight_yawn"]
        self.w_glare   = fc["weight_glare"]

        self.thr_l1 = ac["level_1_score"]
        self.thr_l2 = ac["level_2_score"]
        self.thr_l3 = ac["level_3_score"]
        self.thr_l4 = ac["level_4_score"]

        self.ear_open     = pc["ear_open_threshold"]
        self.ear_closed   = pc["ear_closed_threshold"]
        self.perclos_crit = pc["perclos_critical"]
        self.pitch_crit   = pc["head_pitch_critical"]
        self.yaw_crit     = pc["head_yaw_critical"]
        self.blink_lo     = pc["blink_rate_low"]
        self.blink_hi     = pc["blink_rate_high"]

        self._alert_level   = AlertLevel.NONE
        self._level_entry_t = {}
        self._cooldown_s    = ac["cooldown_s"]
        self._escalation_s  = ac["escalation_delay_s"]

        self._score_ema = 100.0
        self._alpha     = 1.0 / cfg["temporal"]["score_smoothing_frames"]

        self._no_face_start: Optional[float] = None
        self._consec_nf_s   = 0.0

    def _score_ear(self, ear: float) -> float:
        lo, hi = self.ear_closed, self.ear_open
        return max(0.0, min(100.0, (ear - lo) / (hi - lo) * 100.0))

    def _score_perclos(self, perclos: float) -> float:
        return max(0.0, 100.0 - (perclos / self.perclos_crit) * 100.0)

    def _score_blink(self, rate: float) -> float:
        if rate < self.blink_lo:
            return max(0.0, rate / self.blink_lo * 80.0)
        elif rate > self.blink_hi:
            return max(0.0, 100.0 - (rate - self.blink_hi) / self.blink_hi * 60.0)
        return 100.0

    def _score_pitch(self, pitch: float) -> float:
        """Score head pitch deviation (looking up/down)."""
        p = min(abs(pitch) / self.pitch_crit, 1.0)
        return max(0.0, 100.0 - p * 100.0)

    def _score_yaw(self, yaw: float) -> float:
        """Score head yaw deviation (looking left/right)."""
        y = min(abs(yaw) / self.yaw_crit, 1.0)
        return max(0.0, 100.0 - y * 100.0)

    def _score_pose(self, pitch: float, yaw: float) -> float:
        """Deprecated: use _score_pitch and _score_yaw instead."""
        p = min(abs(pitch) / self.pitch_crit, 1.0)
        y = min(abs(yaw)   / self.yaw_crit,   1.0)
        return max(0.0, 100.0 - (0.6 * p + 0.4 * y) * 100.0)

    def _score_yawn_count(self, yawn_count: int) -> float:
        return max(0.0, 100.0 - min(yawn_count, 5) * 18.0)

    def _score_glare(self, glare_severity: float) -> float:
        """
        Convert glare severity (0-100) to attention impact score.
        
        Glare acts as external environmental stressor:
        - No glare (0): Score = 100 (neutral)
        - Mild glare (25): Score = 85 (slight visibility concern)
        - Moderate glare (50): Score = 60 (significant distraction)
        - Severe glare (100): Score = 0 (critical visibility impairment)
        
        Formula: 100 - glare_severity (inverse relationship)
        """
        return max(0.0, 100.0 - glare_severity)

    def _raw_alert_level(self, score: float, pf: PerceptionFrame,
                         tf: TemporalFeatures) -> int:
        if tf.microsleep_count > 0 and tf.microsleep_duration_ms > 800:
            return AlertLevel.EMERGENCY
        if tf.perclos > self.perclos_crit and score < 30:
            return AlertLevel.CRITICAL
        if not pf.face_detected and self._consec_nf_s > 4.0:
            return AlertLevel.WARNING
        if   score <= self.thr_l4: return AlertLevel.EMERGENCY
        elif score <= self.thr_l3: return AlertLevel.CRITICAL
        elif score <= self.thr_l2: return AlertLevel.WARNING
        elif score <= self.thr_l1: return AlertLevel.ADVISORY
        return AlertLevel.NONE

    def _apply_hysteresis(self, raw_level: int) -> int:
        now = time.time()
        if raw_level > self._alert_level:
            self._alert_level = raw_level
            self._level_entry_t[raw_level] = now
        elif raw_level < self._alert_level:
            entry = self._level_entry_t.get(self._alert_level, now)
            if (now - entry) > self._escalation_s:
                self._alert_level = raw_level
        return self._alert_level

    @staticmethod
    def _encode_can(score: float, alert: int, perclos: float) -> str:
        s  = max(0, min(255, int(score * 2.55)))
        al = alert & 0x0F
        pc = max(0, min(255, int(perclos * 255)))
        return f"0x1EF00#{s:02X}{al:02X}{pc:02X}00"

    def process(self, pf: PerceptionFrame, tf: TemporalFeatures,
                gdf: Optional[GlareDetectionFrame] = None) -> FusionOutput:
        now = time.time()
        fo  = FusionOutput(timestamp=now)

        if not pf.face_detected:
            if self._no_face_start is None:
                self._no_face_start = now
            self._consec_nf_s = now - self._no_face_start
        else:
            self._no_face_start = None
            self._consec_nf_s   = 0.0

        s_ear     = self._score_ear(pf.ear_smooth)
        s_perclos = self._score_perclos(tf.perclos)
        s_blink   = self._score_blink(tf.blink_rate)
        s_pitch   = self._score_pitch(pf.pitch_smooth)
        s_yaw     = self._score_yaw(pf.yaw_smooth)
        s_yawn    = self._score_yawn_count(tf.yawn_count_recent)
        s_glare   = self._score_glare(gdf.glare_severity) if gdf else 100.0

        fo.score_ear     = s_ear
        fo.score_perclos = s_perclos
        fo.score_blink   = s_blink
        fo.score_pose    = (s_pitch + s_yaw) / 2.0
        fo.score_yawn    = s_yawn
        fo.score_glare   = s_glare
        if gdf:
            fo.glare_severity = gdf.glare_severity

        total_w = (self.w_ear + self.w_perclos + self.w_blink +
                   self.w_pitch + self.w_yaw + self.w_yawn + self.w_glare)
        raw_score = (
            self.w_ear     * s_ear     +
            self.w_perclos * s_perclos +
            self.w_blink   * s_blink   +
            self.w_pitch   * s_pitch   +
            self.w_yaw     * s_yaw     +
            self.w_yawn    * s_yawn    +
            self.w_glare   * s_glare
        ) / total_w

        if not pf.face_detected:
            raw_score = min(raw_score, 70.0)

        self._score_ema        = self._alpha * raw_score + (1 - self._alpha) * self._score_ema
        fo.attention_score     = round(self._score_ema, 2)
        fo.kss_estimate        = round(attention_to_kss(self._score_ema), 2)
        fo.fatigue_probability = round(fatigue_probability(self._score_ema), 3)

        prev_level = self._alert_level
        raw_level  = self._raw_alert_level(self._score_ema, pf, tf)
        alert_out  = self._apply_hysteresis(raw_level)

        fo.alert_level = alert_out
        fo.alert_label = ALERT_LABELS[alert_out]
        fo.alert_new   = (alert_out != prev_level)

        fo.is_calibrated         = pf.calibrated
        fo.face_present          = pf.face_detected
        fo.consecutive_no_face_s = round(self._consec_nf_s, 1)
        fo.can_frame             = self._encode_can(
            fo.attention_score, alert_out, tf.perclos)

        return fo