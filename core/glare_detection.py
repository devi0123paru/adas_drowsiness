"""
core/glare_detection.py
-----------------------
Module 2: Glare Detection - Oncoming Headlight Detection

Detects glare caused by high-beam headlights from oncoming vehicles using a
simulated Light Dependent Resistor (LDR) sensor. Focuses on rapid, intense
light events that indicate oncoming or surrounding vehicle headlights.

Architecture:
- Simulated LDR sensor generates light intensity (0-255 lux range)
- FAST detection: Responds within 1-2 frames (33-66ms at 30fps)
- Absolute intensity thresholds (not baseline-dependent)
- Severity based on how much light exceeds normal driving conditions

Light Level Interpretation:
- 0-80 lux: Night driving (safe)
- 80-120 lux: Normal daytime/well-lit area
- 120-180 lux: Bright sunlight / street lights (mild warning)
- 180-220 lux: Strong sunlight / approaching headlights (glare warning)
- 220+ lux: Oncoming high-beam or direct sunlight (critical glare)

Future Integration (ready for real Arduino + CAN bus):
- _read_arduino_ldr() : To be replaced with serial port read from Arduino
- _encode_glare_can() : Glare severity packed into extended CAN frame
"""

import time
import math
import numpy as np
import cv2
from dataclasses import dataclass
from collections import deque
from typing import Optional


@dataclass
class GlareDetectionFrame:
    timestamp: float = 0.0
    light_intensity: float = 0.0  # 0-255, simulated LDR value (lux)
    is_glare_detected: bool = False  # True if light exceeds glare threshold
    glare_severity: float = 0.0  # 0-100, based on absolute light intensity
    light_category: str = "SAFE"  # SAFE, MILD, WARN, CRITICAL
    peak_light_recent: float = 0.0  # Highest light in last 30 frames


class GlareDetector:
    """
    Detects glare from oncoming vehicle headlights using FAST thresholds.
    
    Key differences from baseline-tracking approach:
    - Responds immediately to bright light (1-2 frames)
    - Uses absolute intensity thresholds, not relative to baseline
    - Sensitive to oncoming headlights and bright environmental conditions
    - Tracks peak light to detect even brief glare events
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Configuration dict with keys:
                - glare_detection.brightness_warning_threshold: light level for yellow warning
                - glare_detection.brightness_critical_threshold: light level for red alert
                - glare_detection.glare_severity_max: max lux value for 100% severity
                - glare_detection.ldr_simulation_enabled: use simulated data
        """
        self.cfg = cfg
        gc = cfg.get("glare_detection", {})

        # Absolute light intensity thresholds (lux)
        self.warn_threshold = gc.get("brightness_warning_threshold", 120)
        self.critical_threshold = gc.get("brightness_critical_threshold", 180)
        self.max_severity_lux = gc.get("glare_severity_max", 255)
        
        self.ldr_simulation = gc.get("ldr_simulation_enabled", True)
        self.target_fps = cfg.get("system", {}).get("target_fps", 30)

        # Track peak light over short window (recent 30 frames = 1 second at 30fps)
        self.peak_buffer: deque = deque(maxlen=30)
        self._frame_count = 0

        print("[GlareDetector] Initialized - ONCOMING HEADLIGHT DETECTION")
        print(f"  Warning threshold: {self.warn_threshold} lux")
        print(f"  Critical threshold: {self.critical_threshold} lux")
        print(f"  Max severity lux: {self.max_severity_lux} lux")
        print(f"  Simulation mode: {self.ldr_simulation}")
        print(f"  Response time: 1-2 frames (~33-66ms at 30fps)")

    def _read_arduino_ldr(self) -> float:
        """
        Read LDR sensor from Arduino microcontroller.
        
        FUTURE TODO: Implement serial communication to Arduino:
        1. Connect via pyserial on COM port (configurable)
        2. Arduino sends light intensity as 0-255 ADC value
        3. Parse bytes: e.g., "L:175\n" → 175
        4. Add error handling for dropped frames
        
        Returns:
            Light intensity value (0-255 lux)
        """
        # TODO: Add pyserial import and Arduino communication
        # For now, returns 0 - will be replaced in production
        return 0.0

    def _sim_ldr(self) -> float:
        """
        Simulate realistic LDR sensor with rapid light changes.
        
        Models:
        - Normal driving: 40-120 lux fluctuation (time-of-day dependent)
        - Oncoming headlights: sudden 180-255 lux spike (~7% probability per frame)
        - Frequency: ~14 frames between events at 30fps (467ms average)
        
        This is MORE realistic than baseline-approach for detecting headlights.
        """
        t_phase = (time.time() % 86400) / 86400.0  # 0-1 over 24h
        base_light = 50 + 70 * np.sin(2 * np.pi * t_phase - np.pi / 2)

        # Random noise
        noise = np.random.normal(0, 8)

        # ONCOMING HEADLIGHTS: Sudden bright spike (sensitive detection)
        # 7% chance per frame = headlights every ~14 frames (467ms at 30fps)
        if np.random.random() < 0.07:
            # Genuine approaching headlight: bright spike
            headlight_intensity = np.random.uniform(180, 255)
            intensity = min(255, headlight_intensity + noise)
        else:
            # Normal ambient light
            intensity = min(255, max(0, base_light + noise))

        return max(0.0, min(255.0, intensity))

    def _analyze_frame_brightness(self, frame) -> float:
        """
        Analyze actual camera frame to measure light intensity.
        
        Converts frame to HSV, extracts Value (brightness) channel,
        calculates mean brightness, and maps to 0-255 lux scale.
        
        Bright flashlight: mean brightness increases significantly.
        
        Args:
            frame: BGR numpy array from cv2.VideoCapture
            
        Returns:
            Estimated light intensity (0-255 lux)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        try:
            # Convert BGR to HSV to get brightness (Value channel)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2].astype(np.float32)
            
            # Calculate mean brightness (0-255 scale)
            mean_brightness = np.mean(v_channel)
            
            # Map to lux scale (0-255)
            # Empirically: dark frame ~20, normal ~100, flashlight ~200+
            return float(mean_brightness)
        except Exception:
            return 0.0

    def _get_light_intensity(self, frame=None) -> float:
        """Fetch light intensity from frame analysis or simulation."""
        if frame is not None:
            # Use actual frame brightness if available
            return self._analyze_frame_brightness(frame)
        elif self.ldr_simulation:
            return self._sim_ldr()
        else:
            return self._read_arduino_ldr()

    def _categorize_light(self, intensity: float) -> str:
        """
        Categorize light level for UI/logging.
        Categories: SAFE < NORMAL < MILD < WARN < CRITICAL
        """
        if intensity < 80:
            return "SAFE"
        elif intensity < self.warn_threshold:  # 80-120
            return "NORMAL"
        elif intensity < self.critical_threshold:  # 120-180
            return "MILD"
        elif intensity <= self.max_severity_lux * 0.85:  # 180-216 at max 255
            return "WARN"
        else:  # 216+
            return "CRITICAL"

    def _score_glare_severity(self, intensity: float) -> float:
        """
        Convert absolute light intensity to glare severity (0-100 scale).
        
        Uses absolute thresholds, not baseline-dependent:
        - 0 lux → 0% severity (dark night)
        - 120 lux → 0% severity (normal driving)
        - 180 lux → 50% severity (bright lights detected)
        - 255 lux → 100% severity (critical glare)
        
        Args:
            intensity: light intensity (0-255 lux)
            
        Returns:
            Severity score (0-100)
        """
        # Below warning threshold = no glare
        if intensity < self.warn_threshold:
            return 0.0

        # Linear interpolation from warn to critical
        if intensity < self.critical_threshold:
            range_delta = self.critical_threshold - self.warn_threshold
            return (intensity - self.warn_threshold) / range_delta * 50.0

        # Above critical threshold
        range_delta = self.max_severity_lux - self.critical_threshold
        return 50.0 + (intensity - self.critical_threshold) / range_delta * 50.0

    def process(self, frame) -> GlareDetectionFrame:
        """
        Detect oncoming headlight glare from actual camera frame brightness.
        
        FAST DETECTION: Responds in 1-2 frames (~33-66ms).
        - Analyzes frame brightness from HSV Value channel
        - Triggers immediately on bright light (flashlight, headlights)
        - Tracks peak light to catch brief glare events
        
        How it works:
        - When flashlight points at camera: frame brightens → detection triggers
        - Mean brightness 0-100 = safe, 100-180 = warning, 180-255 = critical
        
        Args:
            frame: BGR camera frame from cv2.VideoCapture.
        
        Returns:
            GlareDetectionFrame with:
            - light_intensity: measured frame brightness (0-255)
            - is_glare_detected: True if brightness above warning threshold
            - glare_severity: 0-100 based on absolute brightness
            - light_category: SAFE/NORMAL/MILD/WARN/CRITICAL
            - peak_light_recent: max brightness in last 30 frames
        """
        self._frame_count += 1
        now = time.time()

        # Fetch brightness from actual frame
        light = self._get_light_intensity(frame)
        
        # Track peak light (for detecting brief glare events)
        self.peak_buffer.append(light)
        peak = max(self.peak_buffer) if self.peak_buffer else light

        # Calculate severity (0-100)
        severity = self._score_glare_severity(light)
        
        # Categorize
        category = self._categorize_light(light)
        
        # Detect glare: any light above warning threshold
        is_glare = (light >= self.warn_threshold)

        # Create output frame
        gdf = GlareDetectionFrame(
            timestamp=now,
            light_intensity=round(light, 1),
            is_glare_detected=is_glare,
            glare_severity=round(severity, 1),
            light_category=category,
            peak_light_recent=round(peak, 1),
        )

        return gdf
