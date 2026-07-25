# euro-vision

Computer vision pipeline for automated Euro coin scanning and rare coin detection. Segments individual coins from tray images, normalises orientation, classifies denomination, and flags potentially valuable coins against a curated database of rare Euro issues.

---

## Overview

euro-vision is a personal project aimed at automating the process of scanning large volumes of Euro coins to identify rare or valuable specimens. Coins are placed in a fixed 3D-printed tray, photographed, and processed end-to-end by the pipeline — from raw tray image to a flagged list of potentially valuable coins.

---

## Pipeline

```
Tray Image
    │
    ▼
1. Coin Segmentation       — YOLO-based object detection to locate and crop individual coins
    │
    ▼
2. Normalisation           — Orientation correction and size normalisation via OpenCV
    │
    ▼
3. Denomination Classification  — CNN classifier to identify coin value (1c → €2)
    │
    ▼
4. Rare Coin Detection     — Image similarity / classification against curated rare coin database
    │
    ▼
Flagged Results
```

---

## Features

- Automated segmentation of coins from a controlled tray image
- Rotation-invariant normalisation for consistent coin orientation
- Denomination classification across all Euro coin values
- Rare coin detection against a hand-curated database of ~50 high-value Euro issues
- Focus on €1 and €2 coins where rare variants have the most value

---

## Target Rare Coins

The rare coin database focuses on the most valuable Euro coins likely to appear in circulation, including:

- Low-mintage commemorative 2 euro coins (Monaco, Vatican, San Marino)
- Early issue Greek and Cypriot standard coins (2004–2015)
- Error coins and misstrikes
- Selected national commemoratives with mintage under ~200,000

---

## Tech Stack

| Component | Technology |
|---|---|
| Coin detection | YOLOv8 |
| Image processing | OpenCV |
| Classification | PyTorch / TensorFlow |
| Database | SQLite + local image store |
| Interface | Python CLI / (planned) web UI |

---

## Hardware

Coins are photographed in a custom 3D-printed tray designed to hold coins flat and evenly spaced, allowing consistent overhead imaging across scans.

---

## Status

> 🚧 Work in progress

- [ ] Tray segmentation model
- [ ] Normalisation pipeline
- [ ] Denomination classifier
- [ ] Rare coin database (initial set)
- [ ] Rare coin detection model
- [ ] CLI interface
- [ ] Results export

---

## Motivation

Built as a portfolio project combining computer vision and practical numismatics. The goal is to be able to scan coins obtained from banks or accumulated over time and automatically surface anything worth keeping.
