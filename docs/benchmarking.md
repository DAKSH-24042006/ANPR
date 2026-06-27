# Benchmarking Guide

## Overview

The benchmarking script (`benchmarks/benchmark.py`) measures the performance of the complete ANPR system across the test dataset. It generates machine-readable results (`benchmark_results.json`) and a human-readable report (`benchmark_report.md`).

---

## Benchmarking Strategy

- **Full test set:** When the test dataset contains ≤1000 images, all images are benchmarked.
- **Representative sample:** If the test dataset exceeds 1000 images, a stratified random sample of at least 100 images is used, preserving vehicle category distribution.
- **Reproducibility:** A fixed random seed (`42`) ensures consistent sampling.

The test dataset for this project contains **175 images** — all are benchmarked.

---

## Metrics Collected

### Vehicle Detector (YOLO11s COCO)

| Metric | Source |
| :--- | :--- |
| Average inference time (ms) | Measured per-image during benchmark |
| COCO mAP@50 | Pretrained model paper (0.479) |
| COCO mAP@50-95 | Pretrained model paper (0.325) |

> The vehicle detector uses the pretrained COCO weights without retraining. Detection capability relies on the pretrained model. Latency is benchmarked directly.

### Plate Detector (YOLO11s Fine-Tuned)

| Metric | Source |
| :--- | :--- |
| Average inference time (ms) | Measured per-image during benchmark |
| Precision | Training results.csv (best epoch) |
| Recall | Training results.csv (best epoch) |
| mAP@50 | Training results.csv (best epoch) |
| mAP@50-95 | Training results.csv (best epoch) |

### Best Epoch Validation Metrics (from Training)

| Metric | Value |
| :--- | :--- |
| Epoch | 97 |
| Precision | 0.9209 |
| Recall | 0.8528 |
| mAP@50 | 0.9242 |
| mAP@50-95 | 0.7592 |

### OCR Engine (PP-OCRv5)

| Metric | Description |
| :--- | :--- |
| Average OCR time (ms) | Per-image OCR extraction time |
| Plate Recognition Accuracy (PRA) | Successful OCR / images with plate annotations |
| Character Accuracy Rate (CAR) | Requires human-annotated plate text (not available in YOLO labels) |

> **Note on CAR:** The YOLO-format labels contain only bounding box coordinates, not plate text. Character Accuracy Rate requires human-annotated ground-truth text. PRA is computed instead as the ratio of successful OCR recognitions to annotated plate images.

### Complete System

| Metric | Description |
| :--- | :--- |
| Total inference time (ms) | End-to-end pipeline latency |
| FPS | Frames per second (1000 / total_ms) |
| CPU utilisation (%) | Sampled via psutil during benchmark |
| RAM usage (MB) | Process memory usage |
| GPU VRAM (MB) | PyTorch CUDA memory (if available) |

---

## Running the Benchmark

```bash
cd D:/ANPR
.venv/Scripts/python.exe benchmarks/benchmark.py
```

---

## Output Files

| File | Description |
| :--- | :--- |
| `benchmarks/benchmark_results.json` | Machine-readable metrics |
| `benchmarks/benchmark_report.md` | Human-readable markdown report |

---

## Interpreting Results

### Pipeline Bottleneck Analysis

Typical latency distribution on CPU:

| Stage | % of Total Time |
| :--- | :--- |
| OCR (PP-OCRv5) | ~60-70% |
| Vehicle Detection (YOLO11s) | ~15-25% |
| Plate Detection (YOLO11s) | ~5-10% |
| Image Enhancement | ~1-2% |
| Other stages | <1% |

The OCR engine is the primary bottleneck. GPU acceleration significantly reduces YOLO detection time but has less impact on PaddleOCR latency.
