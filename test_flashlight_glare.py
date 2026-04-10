#!/usr/bin/env python3
"""
test_flashlight_glare.py
------------------------
Test glare detection response to bright light (flashlight simulation)

How to use:
1. Run this script with your camera/webcam visible
2. Point a flashlight at your camera
3. Watch the glare detection respond in real-time
4. Press 'q' to quit

Expected behavior:
- Baseline: light_intensity ~50-100 (normal room)
- Flashlight on: light_intensity ~200-255 (glare detected!)
- HUD Shows: "⚡ HEADLIGHTS!" when severity > 50%
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.glare_detection import GlareDetector
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config/dms_config.yaml")
    glare_detector = GlareDetector(cfg)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try: python test_flashlight_glare.py")
        return
    
    res = cfg["system"]["camera_resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    
    print("\n" + "="*70)
    print(" GLARE DETECTION TEST - Flashlight Response")
    print("="*70)
    print("\nInstructions:")
    print("1. Point a flashlight/torch at your camera")
    print("2. Watch light_intensity number spike ~200-255")
    print("3. Watch glare_severity increase to detect the glare")
    print("4. Press 'q' to quit\n")
    print("Thresholds (from config):")
    print(f"  - Warning: {glare_detector.warn_threshold} lux")
    print(f"  - Critical: {glare_detector.critical_threshold} lux")
    print("="*70 + "\n")
    
    frame_count = 0
    peak_brightness = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run glare detection
            gdf = glare_detector.process(frame)
            
            # Track peak
            if gdf.light_intensity > peak_brightness:
                peak_brightness = gdf.light_intensity
            
            frame_count += 1
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                status = "🔴 GLARE DETECTED!" if gdf.is_glare_detected else "✓ OK"
                print(f"Frame {frame_count:4d} | "
                      f"Intensity: {gdf.light_intensity:3.0f}L | "
                      f"Severity: {gdf.glare_severity:5.1f}% | "
                      f"Category: {gdf.light_category:8s} | "
                      f"{status}")
            
            # Render HUD
            h, w = frame.shape[:2]
            
            # Top status bar
            cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 30), -1)
            
            # Glare status
            if gdf.is_glare_detected:
                if gdf.glare_severity > 50:
                    col = (0, 40, 230)  # Red - critical
                    label = "⚡ FLASHLIGHT DETECTED!"
                else:
                    col = (0, 165, 255)  # Orange - warning
                    label = "⚠ BRIGHT LIGHT"
            else:
                col = (0, 200, 100)  # Green - safe
                label = "○ CLEAR"
            
            cv2.putText(frame, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Brightness: {gdf.light_intensity:.0f}L | "
                               f"Severity: {gdf.glare_severity:.0f}% | "
                               f"{gdf.light_category}",
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Brightness bar
            bar_w = 300
            bar_h = 20
            bar_x, bar_y = w - bar_w - 20, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            fill = int(bar_w * gdf.light_intensity / 255.0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), col, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
            
            # Display
            cv2.imshow("Glare Detection - Point Flashlight at Camera (q=quit)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(f" Test Complete - Processed {frame_count} frames")
        print(f" Peak brightness detected: {peak_brightness:.0f}L")
        print("="*70)


if __name__ == "__main__":
    main()
