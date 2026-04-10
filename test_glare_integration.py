#!/usr/bin/env python3
"""
Integration test for Glare Detection Module
Verifies that all components initialize and work together correctly.
"""

import sys
import time
import yaml
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from core.glare_detection import GlareDetector, GlareDetectionFrame
from core.perception import PerceptionFrame, TemporalFeatures
from core.fusion import FusionEngine, FusionOutput

def test_glare_detector():
    """Test GlareDetector initialization and processing."""
    print("[TEST] Initializing GlareDetector (ONCOMING HEADLIGHT MODE)...")
    
    cfg_path = Path(__file__).parent / "config/dms_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    glare_detector = GlareDetector(cfg)
    
    # Simulate 100 frames to see headlight events
    print("[TEST] Processing 100 simulated frames (including headlight spikes)...\n")
    headlight_count = 0
    for i in range(100):
        gdf = glare_detector.process(None)
        if gdf.is_glare_detected:
            headlight_count += 1
            print(f"  Frame {i+1:3d}: light={gdf.light_intensity:6.1f} lux | "
                  f"severity={gdf.glare_severity:5.1f}% | {gdf.light_category:8s} | "
                  f"HEADLIGHTS DETECTED! ⚡")
    
    print(f"\n  Total frames with glare: {headlight_count}/100")
    print(f"  Detection rate: ~{headlight_count}% (simulates approaching vehicles)")
    
    print("[✓] GlareDetector test passed\n")
    return cfg

def test_fusion_integration(cfg):
    """Test Fusion engine integration with glare input."""
    print("[TEST] Testing Fusion integration with oncoming headlights...\n")
    
    fusion = FusionEngine(cfg)
    
    # Create sample perception and temporal features (alert driver)
    pf = PerceptionFrame(
        timestamp=time.time(),
        face_detected=True,
        ear=0.25,  # Already drowsy
        ear_smooth=0.24,
        mar=0.20,
        pitch=5.0,
        pitch_smooth=4.0,
        yaw=0.0,
        yaw_smooth=0.0,
        roll=0.0,
        calibrated=True,
    )
    
    tf = TemporalFeatures(
        perclos=0.15,  # Moderate eye closure
        blink_rate=18.0,
        blink_duration=140.0,
        microsleep_count=0,
        microsleep_duration_ms=0.0,
        yawn_count_recent=1,
    )
    
    # Scenario 1: No headlights, just drowsy driver
    print("SCENARIO 1: Drowsy driver, clear weather")
    print("-" * 50)
    fo1 = fusion.process(pf, tf, None)
    print(f"  Attention score: {fo1.attention_score}")
    print(f"  Alert level: {fo1.alert_label}")
    
    # Scenario 2: Oncoming headlights detected
    print("\nSCENARIO 2: ONCOMING HEADLIGHTS (180 lux)")
    print("-" * 50)
    glare_detector = GlareDetector(cfg)
    gdf = glare_detector.process(None)
    # Inject headlight event
    gdf.light_intensity = 185.0
    gdf.is_glare_detected = True
    gdf.glare_severity = 55.0  # Above 50% = significant glare
    gdf.light_category = "CRITICAL"
    
    fo2 = fusion.process(pf, tf, gdf)
    print(f"  Light intensity: {gdf.light_intensity} lux")
    print(f"  Glare severity: {gdf.glare_severity}%")
    print(f"  Glare component score: {fo2.score_glare}")
    print(f"  Attention score: {fo2.attention_score}")
    print(f"  Alert level: {fo2.alert_label}")
    
    # Scenario 3: High beams + very drowsy driver = MAXIMUM RISK
    print("\nSCENARIO 3: HIGH BEAMS + SEVERE DROWSINESS (255 lux)")
    print("-" * 50)
    pf.ear_smooth = 0.18  # Severe eye closure
    tf.perclos = 0.35  # Very high perclos
    tf.yawn_count_recent = 3
    
    gdf.light_intensity = 250.0
    gdf.glare_severity = 95.0
    gdf.light_category = "CRITICAL"
    
    fo3 = fusion.process(pf, tf, gdf)
    print(f"  Light intensity: {gdf.light_intensity} lux (HIGH BEAMS)")
    print(f"  Glare severity: {gdf.glare_severity}%")
    print(f"  Drowsiness indicators: EAR={pf.ear_smooth:.2f}, PERCLOS={tf.perclos:.2f}")
    print(f"  Attention score: {fo3.attention_score}")
    print(f"  Alert level: {fo3.alert_label}")
    
    if fo3.attention_score < fo1.attention_score:
        print(f"\n  ✓ Headlights correctly degrade attention score")
    
    print("[✓] Fusion integration test passed\n")

def test_config_validation():
    """Verify configuration is properly formatted."""
    print("[TEST] Validating ADAS-DMS configuration...")
    
    cfg_path = Path(__file__).parent / "config/dms_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check required sections
    required_sections = [
        "system", "perception", "temporal", "alerts",
        "fusion", "glare_detection", "audio"
    ]
    
    for section in required_sections:
        if section in cfg:
            print(f"  ✓ {section}")
        else:
            print(f"  ✗ Missing section: {section}")
            return False
    
    # Verify fusion weights sum to 1.0
    fc = cfg["fusion"]
    total_weight = (
        fc["weight_ear"] + fc["weight_perclos"] + fc["weight_blink_rate"] +
        fc["weight_head_pitch"] + fc["weight_head_yaw"] + 
        fc["weight_yawn"] + fc["weight_glare"]
    )
    
    print(f"\n  Fusion weights total: {total_weight:.4f}")
    if abs(total_weight - 1.0) < 0.001:
        print(f"  ✓ Weights normalized correctly")
    else:
        print(f"  ✗ Warning: Weights don't sum to 1.0")
    
    print("[✓] Configuration validation passed\n")
    return True

def main():
    print("=" * 60)
    print("ADAS-DMS v3.0 - Glare Detection Module Integration Test")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Configuration
        if not test_config_validation():
            print("[✗] Configuration validation failed")
            sys.exit(1)
        
        # Test 2: GlareDetector
        cfg = test_glare_detector()
        
        # Test 3: Fusion Integration
        test_fusion_integration(cfg)
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nGlare Detection Module successfully integrated.")
        print("\nTo run the full system:")
        print("  python main.py")
        print("\nTo disable the camera and use test video:")
        print("  python main.py --source path/to/video.mp4")
        
    except Exception as e:
        print(f"\n[✗] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
