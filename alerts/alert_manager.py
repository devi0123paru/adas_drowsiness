"""
alerts/alert_manager.py
-----------------------
Handles audio alert delivery with cooldown management.
"""

import threading
import time
import queue
import numpy as np
from core.fusion import AlertLevel, FusionOutput

try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)  # 2 channels for stereo
    pygame.mixer.init()
    HAS_AUDIO = True
except Exception as e:
    HAS_AUDIO = False
    print(f"[AlertManager] pygame not available - audio alerts disabled: {e}")

class AlertManager:
    """
    Manages audio alerts for different alert levels.
    
    Alert Patterns:
    - ADVISORY (Level 1): Single gentle beep (yellow) - Mild drowsiness
    - WARNING (Level 2): Double beep (orange) - Moderate drowsiness or glare
    - CRITICAL (Level 3): Triple rapid beeps (red) - Severe drowsiness detected
    - EMERGENCY (Level 4): Fast repeating beeps (red+) - Immediate danger (microsleep)
    """

    PATTERNS = {
        AlertLevel.ADVISORY:  [
            (440, 200, 100),  # Single 440Hz beep - gentle warning
        ],
        AlertLevel.WARNING:   [
            (660, 250, 150),  # Medium 660Hz beep
            (660, 250, 0),
        ],
        AlertLevel.CRITICAL:  [
            (880, 200, 100),  # High 880Hz beep
            (880, 200, 100),
            (880, 200, 0),
        ],
        AlertLevel.EMERGENCY: [
            (1100, 150, 50),  # Urgent 1100Hz rapid pulses
            (1100, 150, 50),
            (1100, 150, 50),
            (1100, 150, 50),
            (1100, 150, 0),
        ],
    }

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self._cooldown_s = cfg["alerts"]["cooldown_s"]
        self._last_t     = {lv: 0.0 for lv in range(5)}
        self._queue      = queue.Queue(maxsize=4)
        self._history    = []

        threading.Thread(
            target=self._audio_worker, daemon=True).start()

    def trigger(self, fo: FusionOutput):
        """
        Trigger alert if conditions are met and cooldown has passed.
        
        Args:
            fo: FusionOutput with alert_level and attention_score
        """
        if fo.alert_level == AlertLevel.NONE:
            return
        
        now  = time.time()
        last = self._last_t[fo.alert_level]
        
        # Check cooldown to avoid alert spam
        if (now - last) < self._cooldown_s:
            return
        
        self._last_t[fo.alert_level] = now
        
        # Log alert event
        alert_event = {
            "timestamp": now,
            "level": fo.alert_level,
            "label": fo.alert_label,
            "score": fo.attention_score,
        }
        self._history.append(alert_event)
        
        # Queue for audio playback
        try:
            self._queue.put_nowait(fo.alert_level)
            print(f"[AlertManager] 🔊 {fo.alert_label} alert triggered (score: {fo.attention_score:.1f})")
        except queue.Full:
            pass

    def _audio_worker(self):
        """Background thread for audio playback queue."""
        while True:
            try:
                level = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if not HAS_AUDIO:
                continue
            
            # Play alert pattern for this level
            pattern = self.PATTERNS.get(level, [])
            for freq, dur_ms, pause_ms in pattern:
                self._beep(freq, dur_ms)
                if pause_ms > 0:
                    time.sleep(pause_ms / 1000.0)

    @staticmethod
    def _beep(freq: int, duration_ms: int):
        """Generate and play a beep sound at specified frequency and duration."""
        if not HAS_AUDIO:
            return
        sr   = 44100
        n    = int(sr * duration_ms / 1000)
        t    = np.linspace(0, duration_ms / 1000, n, False)
        wave = np.sin(2 * np.pi * freq * t)
        
        # Apply envelope (fade in/out to avoid clicks)
        ramp = min(int(sr * 0.01), n // 4)
        env  = np.ones(n)
        env[:ramp]  = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        
        # Scale to audio range and convert to int16
        wave = (wave * env * 28000).astype(np.int16)
        
        # Create stereo sound (duplicate mono to both channels) and ensure C-contiguous
        stereo_wave = np.ascontiguousarray(np.array([wave, wave]).T.astype(np.int16))
        
        try:
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
            time.sleep(duration_ms / 1000.0)
        except Exception as e:
            print(f"[AlertManager] Error playing sound: {e}")

