# Failure Analysis

## Overview

This document evaluates the ANPR system's performance across challenging real-world scenarios. For each failure category, we provide expected success rates, root causes, and suggested improvements for future development.

---

## Failure Categories

### 1. Motion Blur

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~85% |
| Plate Detection Success | ~60% |
| OCR Success | ~30-40% |

**Root Cause:** Motion blur degrades edge contrast in plate characters. YOLO detectors are relatively robust to mild blur, but OCR accuracy drops sharply when characters become smeared.

**Suggested Improvement:**
- Apply deblurring preprocessing (e.g., Wiener filter) before OCR.
- Train a motion-blur-augmented dataset for the plate detector.
- Use a super-resolution model to enhance blurry plate crops.

---

### 2. Low Light / Night

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~75% |
| Plate Detection Success | ~55% |
| OCR Success | ~35-45% |

**Root Cause:** Low illumination reduces contrast between plate characters and background. Headlight glare can overexpose the plate region. CLAHE helps but has limits.

**Suggested Improvement:**
- Train on night-augmented datasets with varied brightness/gamma.
- Apply histogram equalisation before detection.
- Use infrared-capable cameras for dedicated night capture.

---

### 3. Rain / Wet Conditions

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~80% |
| Plate Detection Success | ~55% |
| OCR Success | ~35% |

**Root Cause:** Water droplets on the lens or plate surface cause reflections and partial occlusion of characters. Wet plates may also have reduced contrast.

**Suggested Improvement:**
- Add rain-augmented training data.
- Apply dehazing/defogging preprocessing.
- Use hydrophobic coatings on capture lenses.

---

### 4. Dirty / Damaged Plates

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~95% |
| Plate Detection Success | ~70% |
| OCR Success | ~25-35% |

**Root Cause:** Mud, dust, or physical damage obscures characters. The plate detector can still localise the plate region, but OCR fails when characters are partially or fully covered.

**Suggested Improvement:**
- Augment training data with synthetic dirt overlays.
- Use image inpainting to reconstruct partially visible characters.
- Integrate a plate quality scoring module to flag low-confidence results.

---

### 5. Extreme Viewing Angles

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~85% |
| Plate Detection Success | ~50% |
| OCR Success | ~30% |

**Root Cause:** Perspective distortion makes plate characters appear skewed. The current pipeline includes optional perspective correction, but severe angles remain challenging.

**Suggested Improvement:**
- Train with aggressive perspective augmentation.
- Implement robust four-point perspective transformation.
- Use STN (Spatial Transformer Networks) for learned rectification.

---

### 6. Partial Occlusion

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~80% |
| Plate Detection Success | ~45% |
| OCR Success | ~20-30% |

**Root Cause:** Other vehicles, barriers, or objects partially cover the plate. YOLO may detect the vehicle but fail to localise the partially hidden plate.

**Suggested Improvement:**
- Multi-frame detection across video sequences.
- Train with occlusion-augmented datasets.
- Use attention-based detection heads.

---

### 7. Multiple Vehicles

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~90% (per vehicle) |
| Plate Detection Success | ~70% (best candidate) |
| OCR Success | ~60% (best candidate) |

**Root Cause:** The current pipeline processes only the highest-confidence vehicle detection. When multiple vehicles are present, non-primary vehicles are ignored.

**Suggested Improvement:**
- Process all detected vehicles in parallel.
- Return an array of detections per image.
- Add vehicle tracking for multi-frame association.

---

### 8. No Visible Plate

| Metric | Estimate |
| :--- | :--- |
| Vehicle Detection Success | ~90% |
| Plate Detection Success | 0% (expected) |
| OCR Success | N/A |

**Root Cause:** The vehicle is detected but no plate is visible (rear view, covered plate, paper dealer plates). This is expected behaviour — the system correctly returns `NO_PLATE`.

**Suggested Improvement:**
- No action needed; this is a correctly handled case.
- Future versions could flag "no plate visible" differently from "plate detection failed."

---

## Summary Table

| Scenario | Vehicle Det. | Plate Det. | OCR | Overall |
| :--- | :---: | :---: | :---: | :---: |
| Clear daylight | ~95% | ~90% | ~80% | High |
| Motion blur | ~85% | ~60% | ~35% | Low |
| Low light / Night | ~75% | ~55% | ~40% | Low |
| Rain | ~80% | ~55% | ~35% | Low |
| Dirty plates | ~95% | ~70% | ~30% | Low |
| Extreme angles | ~85% | ~50% | ~30% | Low |
| Partial occlusion | ~80% | ~45% | ~25% | Very Low |
| Multiple vehicles | ~90% | ~70% | ~60% | Medium |
| No visible plate | ~90% | 0% | N/A | Expected |

---

## Key Observations

1. **OCR is the weakest link:** Even when the plate is correctly detected and cropped, OCR accuracy is highly sensitive to image quality.
2. **Vehicle detection is robust:** YOLO11s pretrained on COCO performs well across most conditions.
3. **Enhancement helps but has limits:** CLAHE contrast enhancement improves readability in moderate conditions but cannot recover heavily degraded images.
4. **Clear daylight performance is strong:** Under ideal conditions, the system achieves high end-to-end accuracy.

---

> **Note:** The success rates above are estimates based on qualitative analysis of the test dataset and common ANPR literature benchmarks. Precise rates would require labelled test subsets for each failure category.
