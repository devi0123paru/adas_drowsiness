"""
main.py
-------
ADAS-DMS v3.0 — Entry Point

Run: python main.py
Stop: Press Q in the camera window
"""

import sys
import time
import argparse
import threading

import cv2
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.perception import PerceptionEngine
from core.fusion import FusionEngine, AlertLevel
from core.hud_renderer import HUDRenderer
from core.glare_detection import GlareDetector
from alerts.alert_manager import AlertManager
from session_logs.session_logger import SessionLogger

try:
    from flask import Flask, send_from_directory
    from flask_socketio import SocketIO
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("[WARNING] Flask not installed. Run: pip install flask flask-socketio eventlet")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ADAS_DMS:

    def __init__(self, cfg: dict, no_web: bool = False):
        self.cfg = cfg
        self.no_web = no_web

        print("[ADAS-DMS] Starting subsystems...")
        self.perception = PerceptionEngine(cfg)
        self.glare_detector = GlareDetector(cfg)
        self.fusion = FusionEngine(cfg)
        self.alert_mgr = AlertManager(cfg)
        self.logger = SessionLogger(cfg["system"]["session_dir"])
        self.hud = HUDRenderer(cfg)

        self._fps = float(cfg["system"]["target_fps"])
        self._running = False

        if HAS_FLASK and not no_web and cfg["system"]["enable_web_dashboard"]:
            self._start_web()
        else:
            print("[ADAS-DMS] Web dashboard disabled.")

    def _start_web(self):
        dash_dir = Path(__file__).parent / "dashboard"
        app = Flask(__name__, static_folder=str(dash_dir))
        app.config["SECRET_KEY"] = "adas-dms-secret"
        self.sio = SocketIO(app, cors_allowed_origins="*",
                            async_mode="threading",
                            logger=False, engineio_logger=False)

        @app.route("/")
        def index():
            return send_from_directory(str(dash_dir), "dashboard.html")

        def _run():
            port = self.cfg["system"]["dashboard_port"]
            self.sio.run(app, host="0.0.0.0", port=port,
                         use_reloader=False, log_output=False)

        threading.Thread(target=_run, daemon=True).start()
        print(f"[ADAS-DMS] Dashboard → http://localhost:{self.cfg['system']['dashboard_port']}")

    def _push(self, pf, tf, fo, gdf):
        if not (HAS_FLASK and not self.no_web and hasattr(self, "sio")):
            return
        state = dict(
            score=round(fo.attention_score, 2),
            kss=round(fo.kss_estimate, 2),
            fatigue=round(fo.fatigue_probability, 3),
            alert=fo.alert_level,
            score_ear=round(fo.score_ear, 1),
            score_perclos=round(fo.score_perclos, 1),
            score_blink=round(fo.score_blink, 1),
            score_pose=round(fo.score_pose, 1),
            score_yawn=round(fo.score_yawn, 1),
            score_glare=round(fo.score_glare, 1),
            ear=round(pf.ear_smooth, 3),
            mar=round(pf.mar, 3),
            perclos=round(tf.perclos, 4),
            blink_rate=round(tf.blink_rate, 1),
            blink_dur=round(tf.blink_duration, 0),
            microsleeps=tf.microsleep_count,
            us_dur=round(tf.microsleep_duration_ms, 0),
            yawn_recent=tf.yawn_count_recent,
            pitch=round(pf.pitch_smooth, 1),
            yaw=round(pf.yaw_smooth, 1),
            roll=round(pf.roll, 1),
            gaze_x=round(pf.gaze_x, 3),
            gaze_y=round(pf.gaze_y, 3),
            au_brow=round(pf.au_brow_raise, 1),
            light_intensity=round(gdf.light_intensity, 1),
            glare_severity=round(gdf.glare_severity, 1),
            is_glare=int(gdf.is_glare_detected),
            light_category=gdf.light_category,
            peak_light_recent=round(gdf.peak_light_recent, 1),
            face=pf.face_detected,
            calibrated=pf.calibrated,
            calib_pct=round(self.perception.calibration.progress, 2),
            no_face_s=round(fo.consecutive_no_face_s, 1),
            can_frame=fo.can_frame,
            fps=round(self._fps, 1),
        )
        try:
            self.sio.emit("state", state)
        except Exception:
            pass

    def run(self, source):
        res = self.cfg["system"]["camera_resolution"]
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera: {source}")
            print(" Try --source 1 or check your camera connection.")
            return

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            cap.set(cv2.CAP_PROP_FPS, self.cfg["system"]["target_fps"])

        self._running = True
        prev_t = time.time()

        print("\n[ADAS-DMS] ── System RUNNING ──────────────────")
        print("[ADAS-DMS] Q = quit R = reset S = save log")
        print("[ADAS-DMS] ────────────────────────────────────\n")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    print("[ADAS-DMS] Stream ended.")
                    break

                # 1. Perception
                pf, tf = self.perception.process(frame, self._fps)

                # 2. Glare Detection
                gdf = self.glare_detector.process(frame)

                # 3. Fusion
                fo = self.fusion.process(pf, tf, gdf)

                # 4. Alerts
                self.alert_mgr.trigger(fo)

                # 5. Logging
                self.logger.log_frame(pf, tf, fo, gdf)
                if fo.alert_new:
                    self.logger.log_event("alert_transition", {
                        "level": fo.alert_level,
                        "label": fo.alert_label,
                        "score": round(fo.attention_score, 1),
                    })

                # 5. FPS
                now = time.time()
                self._fps = 0.95 * self._fps + 0.05 / max(now - prev_t, 1e-6)
                prev_t = now

                # 6. HUD
                frame = self.hud.render(
                    frame, pf, tf, fo, gdf,
                    self._fps,
                    self.perception.calibration.progress,
                )

                # 7. Dashboard push
                self._push(pf, tf, fo, gdf)

                # 8. Show window
                cv2.imshow("ADAS-DMS v3.0 | Q=quit R=reset S=save", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.perception._microsleep_cnt = 0
                    self.perception._yawn_recent = 0
                    self.perception._blink_ts.clear()
                    print("[ADAS-DMS] Counters reset.")
                elif key == ord('s'):
                    self.logger.save()

        except KeyboardInterrupt:
            print("\n[ADAS-DMS] Interrupted.")
        finally:
            self._running = False
            cap.release()
            cv2.destroyAllWindows()
            self.perception.release()
            self.logger.close()
            print("[ADAS-DMS] Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="ADAS-DMS v3.0")
    parser.add_argument("--config", default="config/dms_config.yaml")
    parser.add_argument("--source", default="0",
                        help="Camera index (0,1,...) or video file path")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable browser dashboard")
    args = parser.parse_args()

    cfg_path = Path(__file__).parent / args.config
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_config(str(cfg_path))
    source = int(args.source) if args.source.isdigit() else args.source

    system = ADAS_DMS(cfg, no_web=args.no_web)
    system.run(source)


if __name__ == "__main__":
    main()