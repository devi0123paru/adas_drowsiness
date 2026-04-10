import csv
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from core.fusion import FusionOutput
from core.perception import PerceptionFrame, TemporalFeatures
from core.glare_detection import GlareDetectionFrame


class SessionLogger:

    HEADERS = [
        "timestamp_s", "frame",
        "face_detected",
        "ear", "ear_smooth", "mar", "eye_closed",
        "pitch", "yaw", "roll",
        "gaze_x", "gaze_y",
        "perclos", "blink_rate", "blink_duration_ms",
        "microsleep_count", "microsleep_dur_ms",
        "yawn_count_recent",
        "light_intensity", "glare_severity", "is_glare_detected", "light_category", "peak_light_recent",
        "score_ear", "score_perclos", "score_blink", "score_pose", "score_yawn", "score_glare",
        "attention_score", "kss_estimate", "fatigue_probability",
        "alert_level", "alert_label",
    ]

    def __init__(self, session_dir: str):
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(session_dir) / ts

        self.json_path = base.with_suffix(".json")
        self.csv_path = Path(str(base) + "_telemetry.csv")

        self._events = []
        self._start_t = time.time()
        self._frame_n = 0
        self._lock = threading.Lock()

        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self.HEADERS)
        self._csv_w.writeheader()

        print(f"[Logger] CSV → {self.csv_path}")
        print(f"[Logger] JSON → {self.json_path}")

    def log_frame(self, pf: PerceptionFrame,
                  tf: TemporalFeatures,
                  fo: FusionOutput,
                  gdf: GlareDetectionFrame):
        self._frame_n += 1
        row = {
            "timestamp_s": round(fo.timestamp - self._start_t, 3),
            "frame": self._frame_n,
            "face_detected": int(pf.face_detected),
            "ear": round(pf.ear, 4),
            "ear_smooth": round(pf.ear_smooth, 4),
            "mar": round(pf.mar, 4),
            "eye_closed": int(pf.eye_closed),
            "pitch": round(pf.pitch, 2),
            "yaw": round(pf.yaw, 2),
            "roll": round(pf.roll, 2),
            "gaze_x": round(pf.gaze_x, 4),
            "gaze_y": round(pf.gaze_y, 4),
            "perclos": round(tf.perclos, 4),
            "blink_rate": round(tf.blink_rate, 2),
            "blink_duration_ms": round(tf.blink_duration, 1),
            "microsleep_count": tf.microsleep_count,
            "microsleep_dur_ms": round(tf.microsleep_duration_ms, 1),
            "yawn_count_recent": tf.yawn_count_recent,
            "light_intensity": round(gdf.light_intensity, 1),
            "glare_severity": round(gdf.glare_severity, 1),
            "is_glare_detected": int(gdf.is_glare_detected),
            "light_category": gdf.light_category,
            "peak_light_recent": round(gdf.peak_light_recent, 1),
            "score_ear": round(fo.score_ear, 2),
            "score_perclos": round(fo.score_perclos, 2),
            "score_blink": round(fo.score_blink, 2),
            "score_pose": round(fo.score_pose, 2),
            "score_yawn": round(fo.score_yawn, 2),
            "score_glare": round(fo.score_glare, 2),
            "attention_score": round(fo.attention_score, 2),
            "kss_estimate": round(fo.kss_estimate, 2),
            "fatigue_probability": round(fo.fatigue_probability, 4),
            "alert_level": fo.alert_level,
            "alert_label": fo.alert_label,
        }
        with self._lock:
            self._csv_w.writerow(row)

    def log_event(self, event_type: str, data: dict):
        with self._lock:
            self._events.append({
                "t": round(time.time() - self._start_t, 3),
                "type": event_type,
                **data,
            })

    def save(self):
        with self._lock:
            self._csv_f.flush()
        summary = {
            "session_start_epoch": self._start_t,
            "duration_s": round(time.time() - self._start_t, 1),
            "total_frames": self._frame_n,
            "events": self._events,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Logger] Session saved — {self._frame_n} frames logged.")

    def close(self):
        self.save()
        self._csv_f.close()