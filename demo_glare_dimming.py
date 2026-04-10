#!/usr/bin/env python3
"""
demo_glare_dimming.py
---------------------
Intelligent glare reduction demonstration

How it works:
1. Detects bright regions (headlights/flashlights) in frame
2. Creates mask only over those bright areas
3. Reduces brightness in masked regions
4. Rest of scene remains visible for driver

This protects the driver's vision from bright glare without
blocking the entire view like global dimming would.
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


def reduce_glare(img, glare_severity):
    """
    Intelligent glare reduction - dim only bright regions.
    
    Keeps the rest of the scene visible while reducing the
    intensity of headlights/flashlights.
    """
    if glare_severity <= 0:
        return img
    
    try:
        # Convert to HSV to work with brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v_channel = hsv[:, :, 2]
        
        # Mask bright regions (where headlights are)
        bright_threshold = 180
        mask = (v_channel > bright_threshold).astype(np.float32)
        
        # Smooth the mask for natural transition
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / 255.0
        
        # Reduction: at 50% severity reduce by 30%, at 100% by 60%
        reduction = 0.3 + (glare_severity / 100.0) * 0.3
        
        # Apply reduction only to masked (bright) areas
        v_reduced = v_channel * (1.0 - mask * reduction)
        v_reduced = np.clip(v_reduced, 0, 255)
        
        hsv[:, :, 2] = v_reduced
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    except Exception:
        return img


def main():
    cfg = load_config("config/dms_config.yaml")
    glare_detector = GlareDetector(cfg)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    res = cfg["system"]["camera_resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    
    print("\n" + "="*70)
    print(" SMART GLARE REDUCTION DEMONSTRATION")
    print("="*70)
    print("\nIntelligent local dimming - reduces only bright regions:")
    print("  • Detects bright spots (headlights/flashlight)")
    print("  • Reduces brightness only in those areas")
    print("  • Preserves rest of scene for driver visibility")
    print("  • Smooth transition with Gaussian blur")
    print("\nCompare:")
    print("  • Before: Bright headlights cause glare")
    print("  • After: Headlights dimmed, rest of view clear")
    print("\nPress 'q' to quit")
    print("="*70 + "\n")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect glare
            gdf = glare_detector.process(frame)
            
            # Create split-screen comparison
            h, w = frame.shape[:2]
            
            # Left side: Original frame
            frame_original = frame.copy()
            
            # Right side: With glare reduction applied
            frame_reduced = reduce_glare(frame.copy(), gdf.glare_severity)
            
            # Combine into split view
            combined = np.hstack([frame_original, frame_reduced])
            
            # Draw dividing line
            cv2.line(combined, (w, 0), (w, h), (100, 100, 100), 2)
            
            # Labels
            cv2.putText(combined, "ORIGINAL (Bright Glare)", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, "REDUCED (Smart Dimming)", (w + 20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2, cv2.LINE_AA)
            
            # Status bar
            if gdf.is_glare_detected:
                if gdf.glare_severity > 50:
                    status_text = f"⚡ CRITICAL GLARE: {gdf.glare_severity:.0f}%"
                    status_color = (0, 40, 230)
                else:
                    status_text = f"⚠️  GLARE DETECTED: {gdf.glare_severity:.0f}%"
                    status_color = (0, 165, 255)
            else:
                status_text = "✓ NO GLARE - Safe driving conditions"
                status_color = (0, 200, 100)
            
            # Bottom info panel
            cv2.rectangle(combined, (0, h - 80), (combined.shape[1], h), (20, 20, 30), -1)
            cv2.putText(combined, status_text, (20, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            cv2.putText(combined, f"Brightness: {gdf.light_intensity:.0f}L  |  "
                              f"Severity: {gdf.glare_severity:.0f}%  |  "
                              f"Category: {gdf.light_category}",
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Display
            cv2.imshow("Smart Glare Reduction - Original vs Reduced (q=quit)", combined)
            
            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Processed {frame_count} frames - Demonstration complete")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
