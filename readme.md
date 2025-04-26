# 🌵 AZVision: Computer Vision Application for Beverage Quality Control

[![Build Status](https://img.shields.io/github/actions/workflow/status/jharnos/azvision/ci.yml?branch=main)](https://github.com/jharnos/azvision/actions)
[![License](https://img.shields.io/github/license/jharnos/azvision)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/model-YOLOv8-brightgreen)](https://github.com/ultralytics/ultralytics)
[![Issues](https://img.shields.io/github/issues/jharnos/azvision)](https://github.com/jharnos/azvision/issues)
[![Last Commit](https://img.shields.io/github/last-commit/jharnos/azvision)](https://github.com/jharnos/azvision/commits/main)
[![Made With Love](https://img.shields.io/badge/Made%20with-%F0%9F%92%96-red.svg)](#)

Welcome to **VisionFlow**, a powerful computer vision system designed for internal use at a leading (but anonymous 👀) beverage company. VisionFlow ensures consistent quality, detects visual anomalies in production, and assists with inventory monitoring — all with the magic of machine learning and real-time video processing.

> ⚠️ This repository is for demonstration purposes. Brand-specific content and assets have been anonymized.

---

## 🔍 Features

- **Real-Time Object Detection**
  - Detect bottles, cans, labels, and caps on the production line
- **Anomaly Detection**
  - Flag missing labels, cap misalignment, and fill level inconsistencies
- **Inventory Insight**
  - Count units, track SKUs, and generate visual logs for auditing
- **Performance Dashboard**
  - Live metrics with alerts and historical tracking (optional web UI)

---

## 🛠️ Tech Stack

| Layer                | Technology           |
| -------------------- | -------------------- |
| Language             | Python               |
| Model Framework      | PyTorch / TensorFlow |
| Detection Model      | YOLOv8               |
| Backend              | FastAPI              |
| Stream Handling      | OpenCV               |
| Dashboard (optional) | Streamlit or React   |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/visionflow.git
cd visionflow
pip install -r requirements.txt
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/azvision.git
cd azvision
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Usage

1. Run the application:
```bash
python -m azvision
```

2. Select your camera and resolution from the settings panel.

3. Adjust edge detection or color detection settings as needed.

4. Use the camera calibration tool to set up accurate measurements.

5. Capture images and generate DXF files for CNC machining.

## 📦 Camera Setup

- The application supports both DirectShow and FFmpeg cameras
- Common resolutions (640x480, 1280x720, 1920x1080, 2560x1440) are supported
- Camera settings can be adjusted in real-time

## 📦 DXF Generation

- Contours are automatically detected and simplified
- The DXF file is generated with proper scaling based on calibration
- Output is in inches with decimal precision
- Contours are closed polylines suitable for CNC machining

## 📦 Development

The application is structured into several modules:

- `azvision/gui/`: GUI components and main application
- `azvision/utils/`: Utility functions for image processing and camera handling
- `azvision/calibration/`: Camera calibration functionality
- `azvision/config.py`: Configuration settings

## 📦 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📦 License

This project is licensed under the MIT License - see the LICENSE file for details.
