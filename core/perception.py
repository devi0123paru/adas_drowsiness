"""
core/perception.py
------------------
Extracts every signal from the driver's face each frame.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List
import time

LEFT_EYE        = [362, 385, 387, 263, 373, 380]
RIGHT_EYE       = [33,  160, 158, 133, 153, 144]
LEFT_IRIS       = [474, 475, 476, 477]
RIGHT_IRIS      = [469, 470, 471, 472]
MOUTH_OUTER     = [61, 291, 13, 14, 17, 0, 78, 308]
LEFT_BROW_IDX   = [336, 296, 334, 293, 300]
RIGHT_BROW_IDX  = [107,  66, 105,  63,  70]
NOSE_TIP        = 4
CHIN            = 152
L_EYE_CORNER    = 263
R_EYE_CORNER    = 33
L_MOUTH         = 61
R_MOUTH         = 291

FACE_3D_MODEL = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)

POSE_LM_IDX = [NOSE_TIP, CHIN, L_EYE_CORNER, R_EYE_CORNER, L_MOUTH, R_MOUTH]


@dataclass
class PerceptionFrame:
    timestamp: float = 0.0
    face_detected: bool = False
    ear_left: float  = 0.30
    ear_right: float = 0.30
    ear: float       = 0.30
    ear_smooth: float= 0.30
    eye_closed: bool = False
    mar: float = 0.0
    yawn_detected: bool = False
    pitch: float = 0.0
    yaw:   float = 0.0
    roll:  float = 0.0
    pitch_smooth: float = 0.0
    yaw_smooth:   float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    au_brow_raise:  float = 0.0
    au_brow_furrow: float = 0.0
    calibrated: bool = False


@dataclass
class TemporalFeatures:
    perclos: float        = 0.0
    blink_rate: float     = 0.0
    blink_duration: float = 0.0
    microsleep_count: int = 0
    microsleep_duration_ms: float = 0.0
    yawn_count_recent: int = 0


class KalmanScalar:
    def __init__(self, Q: float = 1e-3, R: float = 0.05):
        self.Q = Q
        self.R = R
        self.P = 1.0
        self.x = None

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return z
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x


class DriverCalibration:
    CALIB_FRAMES = 90

    def __init__(self):
        self._ear_samples:   List[float] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples:   List[float] = []
        self.ear_baseline   = 0.28
        self.pitch_baseline = 0.0
        self.yaw_baseline   = 0.0
        self.done = False

    @property
    def progress(self) -> float:
        return min(len(self._ear_samples) / self.CALIB_FRAMES, 1.0)

    def feed(self, pf: PerceptionFrame):
        if self.done or not pf.face_detected or pf.ear < 0.15:
            return
        self._ear_samples.append(pf.ear)
        self._pitch_samples.append(pf.pitch)
        self._yaw_samples.append(pf.yaw)
        if len(self._ear_samples) >= self.CALIB_FRAMES:
            self.ear_baseline   = float(np.median(self._ear_samples))
            self.pitch_baseline = float(np.median(self._pitch_samples))
            self.yaw_baseline   = float(np.median(self._yaw_samples))
            self.done = True

    def adjusted_ear_threshold(self, base_threshold: float) -> float:
        if not self.done:
            return base_threshold
        return base_threshold * (self.ear_baseline / 0.28)


class PerceptionEngine:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        pc = cfg["perception"]
        tc = cfg["temporal"]

        self.ear_closed_thr = pc["ear_closed_threshold"]
        self.mar_yawn_thr   = pc["mar_yawn_threshold"]
        self.ms_thr_ms      = pc["microsleep_threshold_ms"]

        mp_fm = mp.solutions.face_mesh
        self.fm = mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

        self._kf_ear   = KalmanScalar(Q=5e-4, R=0.03)
        self._kf_pitch = KalmanScalar(Q=0.01, R=0.50)
        self._kf_yaw   = KalmanScalar(Q=0.01, R=0.50)

        win = int(tc["perclos_window_s"] * 30)
        self._closed_buf = deque(maxlen=win)
        self._blink_ts   = deque(maxlen=300)
        self._blink_dur  = deque(maxlen=100)

        self._eye_was_open  = True
        self._close_start_t = None
        self._microsleep_cnt = 0

        self._yawn_cooldown = 0
        self._yawn_recent   = 0
        self._yawn_reset_t  = time.time()

        self.calibration = DriverCalibration()

    def _ear(self, lm, indices, w, h) -> float:
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in indices])
        A = distance.euclidean(pts[1], pts[5])
        B = distance.euclidean(pts[2], pts[4])
        C = distance.euclidean(pts[0], pts[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def _mar(self, lm, w, h) -> float:
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in MOUTH_OUTER])
        A = distance.euclidean(pts[2], pts[6])
        B = distance.euclidean(pts[3], pts[7])
        C = distance.euclidean(pts[0], pts[1]) + 1e-6
        return (A + B) / (2.0 * C)

    def _head_pose(self, lm, w, h) -> Tuple[float, float, float]:
        face_2d = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in POSE_LM_IDX],
            dtype=np.float64
        )
        focal  = w
        cam    = np.array([[focal, 0, w/2],
                           [0, focal, h/2],
                           [0,     0,   1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, cam, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _    = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return angles[0] * 360, angles[1] * 360, angles[2] * 360

    def _gaze(self, lm, w, h) -> Tuple[float, float]:
        def offset(iris_idx, eye_idx):
            iris_cx = np.mean([lm[i].x * w for i in iris_idx])
            iris_cy = np.mean([lm[i].y * h for i in iris_idx])
            pts     = np.array([[lm[i].x * w, lm[i].y * h] for i in eye_idx])
            eye_cx  = np.mean(pts[:, 0])
            eye_cy  = np.mean(pts[:, 1])
            eye_w   = distance.euclidean(pts[0], pts[3]) + 1e-6
            return (iris_cx - eye_cx) / eye_w, (iris_cy - eye_cy) / eye_w
        lx, ly = offset(LEFT_IRIS,  LEFT_EYE)
        rx, ry = offset(RIGHT_IRIS, RIGHT_EYE)
        return (lx + rx) / 2, (ly + ry) / 2

    def _brow_au(self, lm, w, h) -> Tuple[float, float]:
        def brow_height(brow_idx, eye_idx):
            return (np.mean([lm[i].y * h for i in eye_idx]) -
                    np.mean([lm[i].y * h for i in brow_idx]))
        raise_val  = (brow_height(LEFT_BROW_IDX,  LEFT_EYE) +
                      brow_height(RIGHT_BROW_IDX, RIGHT_EYE)) / 2.0
        furrow_val = abs(lm[LEFT_BROW_IDX[2]].x * w -
                         lm[RIGHT_BROW_IDX[2]].x * w)
        return raise_val, furrow_val

    def _update_blink(self, is_closed: bool):
        now = time.time()
        if self._yawn_cooldown > 0:
            self._yawn_cooldown -= 1
        if self._eye_was_open and is_closed:
            self._close_start_t = now
        elif not self._eye_was_open and not is_closed:
            if self._close_start_t is not None:
                dur_ms = (now - self._close_start_t) * 1000
                if dur_ms >= self.ms_thr_ms:
                    self._microsleep_cnt += 1
                elif dur_ms >= 50:
                    self._blink_ts.append(now)
                    self._blink_dur.append(dur_ms)
                self._close_start_t = None
        self._eye_was_open = not is_closed
        if now - self._yawn_reset_t > self.cfg["temporal"]["yawn_decay_window_s"]:
            self._yawn_recent  = 0
            self._yawn_reset_t = now

    def _temporal(self) -> TemporalFeatures:
        now = time.time()
        perclos = sum(self._closed_buf) / max(len(self._closed_buf), 1)
        window  = self.cfg["temporal"]["blink_rate_window_s"]
        recent  = [t for t in self._blink_ts if now - t <= window]
        elapsed = (now - recent[0]) if len(recent) > 1 else window
        blink_rate = len(recent) / elapsed * 60 if recent else 0.0
        blink_dur  = float(np.mean(self._blink_dur)) if self._blink_dur else 0.0
        us_dur = 0.0
        if self._close_start_t is not None:
            us_dur = (now - self._close_start_t) * 1000
        return TemporalFeatures(
            perclos=perclos,
            blink_rate=blink_rate,
            blink_duration=blink_dur,
            microsleep_count=self._microsleep_cnt,
            microsleep_duration_ms=us_dur,
            yawn_count_recent=self._yawn_recent,
        )

    def process(self, frame: np.ndarray, fps: float = 30.0
                ) -> Tuple[PerceptionFrame, TemporalFeatures]:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.fm.process(rgb)
        pf   = PerceptionFrame(timestamp=time.time())

        if res.multi_face_landmarks:
            pf.face_detected = True
            lm = res.multi_face_landmarks[0].landmark

            pf.ear_left  = self._ear(lm, LEFT_EYE,  w, h)
            pf.ear_right = self._ear(lm, RIGHT_EYE, w, h)
            pf.ear       = (pf.ear_left + pf.ear_right) / 2.0
            pf.ear_smooth = self._kf_ear.update(pf.ear)

            thr = self.calibration.adjusted_ear_threshold(self.ear_closed_thr)
            pf.eye_closed = pf.ear_smooth < thr

            pf.mar = self._mar(lm, w, h)
            if self._yawn_cooldown == 0 and pf.mar > self.mar_yawn_thr:
                pf.yawn_detected = True
                self._yawn_recent   += 1
                self._yawn_cooldown  = int(fps * 3)

            pf.pitch, pf.yaw, pf.roll = self._head_pose(lm, w, h)
            pf.pitch_smooth = self._kf_pitch.update(pf.pitch)
            pf.yaw_smooth   = self._kf_yaw.update(pf.yaw)

            try:
                pf.gaze_x, pf.gaze_y = self._gaze(lm, w, h)
            except Exception:
                pass

            pf.au_brow_raise, pf.au_brow_furrow = self._brow_au(lm, w, h)

            self._closed_buf.append(int(pf.eye_closed))
            self._update_blink(pf.eye_closed)

            self.calibration.feed(pf)
            pf.calibrated = self.calibration.done
        else:
            self._closed_buf.append(0)

        tf = self._temporal()
        return pf, tf

    def release(self):
        self.fm.close()