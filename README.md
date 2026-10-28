# SafeEdge AI — Construction Safety Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![UI](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Model%20License-MIT-orange.svg)](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection)
[![Tests](https://img.shields.io/badge/Tests-137%20Passed-brightgreen.svg)]()

**SafeEdge AI** is a professional, unified construction-site safety intelligence platform. It integrates computer-vision PPE compliance monitoring with real-time multi-hazard environmental risk simulation, historical trend analytics, and multi-format audit report generation.

---

## 1. System Architecture & Modules

The platform is organized into five unified modules accessible through an intuitive sidebar navigation:

```
                            SafeEdge AI Platform
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
┌──────────────┐              ┌──────────────┐              ┌────────────────┐
│  AI Computer │              │ Environmental│              │   Executive    │
│  Vision Core │              │  Simulator   │              │   Overview     │
└──────┬───────┘              └──────┬───────┘              └───────┬────────┘
       │                             │                              │
 ┌─────┴──────────┐            ┌─────┴──────────┐                   │
 │ 📷 Image       │            │ 🏗️ Hazard      │                   │
 │   Inspection   │            │   Cockpit      │                   │
 ├────────────────┤            ├────────────────┤                   │
 │ 🎬 Video       │            │ 📊 History &   │                   │
 │   Analytics    │            │   Audit Reports│                   │
 └────────────────┘            └────────────────┘                   │
       │                             │                              │
       └─────────────────────────────┴──────────────────────────────┘
```

### 1. 🏢 Executive Safety Overview
- Cross-system executive dashboard unifying real-time computer vision metrics with environmental risk telemetry.
- System readiness indicator and model status (`Hansung-Cho/yolov8-ppe-detection`).
- Site PPE compliance summary (`Compliant %`, `Violations`, `Active Workers`).
- Live environmental site safety status (`GREEN Normal`, `ORANGE Warning`, `RED Danger`).
- Recent PPE violations feed and recent hazard state-transition log.
- Quick-launch navigation buttons for each system module.
- Prominent dual-engine disclaimers.

### 2. 📷 PPE Image Detection
- Single-frame high-precision PPE inspection for construction workers.
- Supports file uploads (`.jpg`, `.jpeg`, `.png`, `.webp`) and one-click demo loading (`data/sample_workers.jpg`).
- Bounding box annotations, worker identification, and spatial compliance association.
- Detailed compliance breakdown table: Worker ID, Overall Status, Hardhat, Safety Vest, Confidence, and Diagnostic Explanation.
- Detected classes breakdown (`Person`, `Hardhat`, `Safety Vest`, `NO-Hardhat`, `NO-Safety Vest`).
- Downloadable annotated inspection image (`.jpg`).

### 3. 🎬 PPE Video Detection
- Multi-frame temporal worker tracking across video footage (`.mp4`, `.avi`, `.mov`).
- Configurable frame sampling stride (Stride 1 to 15 for optimal CPU throughput).
- Video metadata validation (resolution, duration, FPS, frames, size up to 500 MB).
- Real-time progress bar and ETA feedback during processing.
- Annotated video playback with persistent worker bounding boxes and status banners.
- Safety compliance timeline line chart and frame observation event logs with filtering.
- Downloadable annotated video (MP4), compliance audit log (CSV), and summary report (JSON).

### 4. 🏗️ Environmental Hazard Simulator
- Five-parameter real-time construction site risk cockpit:
  1. **Temperature** (°C) — Hypothermia to heat-stroke risks.
  2. **Relative Humidity** (%) — Electrostatic discharge to heat-stress index.
  3. **Dust Concentration** (µg/m³ PM10) — Respiratory occupational exposure.
  4. **Noise Level** (dB) — Permanent hearing damage / NIOSH acoustic thresholds.
  5. **Hazardous Gas Concentration** (ppm) — Toxic / combustible atmospheric vapor index.
- Dual operating modes: **Manual Slider Control** and **Automatic Bounded Random Walk Drift**.
- Simulator controls: **Start**, **Pause**, **Resume**, **Reset**, and **Step Once**.
- 6 Demonstration scenario presets (**All Safe**, **Dust & Noise**, **Gas Leak Danger**, **Heat Stress**, **Multi-Hazard**, **Reset Defaults**).
- Root-cause diagnostic explanations and Phase 2 recommended protective actions.
- Prominent simulation disclaimer banner: `SIMULATION MODE — NOT LIVE SENSOR DATA`.

### 5. 📊 Hazard History & Reports
- In-memory rolling buffer retaining session telemetry and transition events.
- **Categorized Time-Series Trend Charts**:
  - *Microclimate*: Temperature (°C) & Relative Humidity (%)
  - *Particulates & Acoustics*: Dust (µg/m³) & Noise (dB)
  - *Gas Index*: Hazardous gas concentration trend (ppm)
- **Filtered Historical Readings Table**: Filter by overall status (`Safe`, `Warning`, `Danger`) and row limit.
- **State-Transition Timeline**: Logs escalations and recoveries (`GREEN ➔ ORANGE`, `ORANGE ➔ RED`, `RED ➔ GREEN`, etc.) with timestamps, values, thresholds, reasons, and actions.
- **Telemetry Summary Statistics**: Min, Max, and Mean for all 5 parameters, session duration, and event counts.
- **Multi-Format Export Center**:
  - 📥 **Readings CSV** (`safeedge_sim_readings_<timestamp>.csv`)
  - 📥 **Hazard Events CSV** (`safeedge_sim_events_<timestamp>.csv`)
  - 📥 **Structured JSON Audit Report** (`safeedge_sim_audit_report_<timestamp>.json`)

---

## 2. Verified Model Provenance & Requirements

A generic object detection model (such as COCO YOLO) cannot detect hardhats or safety vests. SafeEdge strictly validates model weights on loading to ensure genuine PPE capabilities.

| Attribute | Specification |
|:---|:---|
| **Model Repository** | [`Hansung-Cho/yolov8-ppe-detection`](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) |
| **Model File** | `models/best.pt` (~6.25 MB) |
| **Base Architecture** | Ultralytics YOLOv8n (Nano) |
| **Framework** | PyTorch >= 2.0.0, Ultralytics >= 8.1.0 |
| **License** | **MIT License** |
| **Required Classes** | `Person`, `Hardhat`, `Safety Vest` |
| **Negative Classes** | `NO-Hardhat`, `NO-Safety Vest` |

---

## 3. Configuration Files

1. **`config/settings.json`**:
   - Model path, device preference (`auto`, `cpu`, `cuda`).
   - Default confidence threshold (`0.35`).
   - Video upload limits (`500 MB`) and default frame stride (`2`).
   - Spatial association heuristics and color hex palettes.
2. **`config/environmental_thresholds.json`**:
   - Physical parameter bounds, units, step increments, and defaults.
   - Safe ranges, warning thresholds, and danger thresholds.
   - Regulatory disclaimer text.

---

## 4. Supported Formats

| Category | Supported Formats |
|:---|:---|
| **Input Images** | `.jpg`, `.jpeg`, `.png`, `.webp` |
| **Input Videos** | `.mp4`, `.avi`, `.mov` (max 500 MB) |
| **Exported Videos** | `.mp4` (H.264 annotated video) |
| **Exported Reports** | `.csv` (RFC 4180 compliant), `.json` (UTF-8 formatted) |

---

## 5. Project Directory Structure

```text
SafeEdge-PPE-Vision/
├── app.py                             # Unified SafeEdge AI Platform entry point
├── environmental_app.py               # Standalone Environmental Simulator entry point
├── main.py                            # CLI image inspection tool
├── requirements.txt                   # Production Python dependencies
├── pytest.ini                         # Pytest test configuration
├── config/
│   ├── settings.json                  # PPE vision model & video configuration
│   └── environmental_thresholds.json  # 5-parameter hazard thresholds & ranges
├── core/
│   ├── __init__.py                    # Core exports
│   ├── model_loader.py                # Model validation & class verification
│   ├── ppe_detector.py                # YOLO inference & spatial filtering
│   ├── ppe_assessor.py                # Rule-based worker compliance engine
│   ├── visualizer.py                  # High-contrast annotation visualizer
│   ├── tracker.py                     # Temporal worker tracker (IoU + centroid)
│   ├── video_processor.py             # Frame-by-frame video processing engine
│   ├── environmental_simulator.py     # 5-parameter stochastic simulation engine
│   ├── hazard_assessment.py           # Environmental diagnostic & safety actions
│   ├── hazard_history.py              # In-memory history, stats & report exports
│   └── environmental_dashboard.py     # Streamlit rendering components (A-H)
├── data/
│   ├── sample_workers.jpg             # Demonstration site photo (3 workers)
│   └── Video Project.mp4              # Demonstration site video footage
├── models/
│   ├── best.pt                        # Verified YOLOv8n PPE model weights
│   ├── download_model.py              # HuggingFace weights downloader
│   └── README.md                      # Model provenance & class documentation
├── output/                            # Output directory for annotated media & reports
└── tests/
    ├── test_model_loader.py           # 6 model validation tests
    ├── test_ppe_detector.py           # 7 PPE detection tests
    ├── test_ppe_assessor.py           # 10 compliance assessment tests
    ├── test_visualizer.py             # 5 visualization tests
    ├── test_tracker.py                # 4 worker tracking tests
    ├── test_video_processor.py        # 8 video processing tests
    ├── test_real_model_integration.py # 2 real weights inference tests
    ├── test_environmental_simulator.py# 30 Phase 1 simulator tests
    ├── test_hazard_assessment.py      # 24 Phase 2 diagnostic tests
    ├── test_hazard_history.py         # 12 Phase 3 history & deduplication tests
    ├── test_hazard_reporting.py       # 20 Phase 4 reports & export tests
    └── test_safeedge_integration.py   # 9 Phase 5 unified platform tests
```

---

## 6. Installation & Setup Guide

### Windows Setup

1. **Clone or Open Repository**:
   ```powershell
   cd "D:\Antigravity\safety of construciton ppe detection\SafeEdge-PPE-Vision"
   ```

2. **Verify / Create Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Model Weights**:
   Ensure `models/best.pt` (~6.25 MB) is present. If missing, download it:
   ```powershell
   python models/download_model.py
   ```

---

## 7. Running the Application

### A. Unified SafeEdge AI Platform (Recommended)
Launches the full platform containing Executive Overview, PPE Image Detection, PPE Video Detection, Environmental Hazard Simulator, and Hazard History & Reports:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### B. Standalone Environmental Hazard Simulator
Launches the dedicated Environmental Hazard Simulator and Report Generator:

```powershell
.\.venv\Scripts\python.exe -m streamlit run environmental_app.py
```

### C. Command-Line PPE Image Inspection
```powershell
# Analyze sample image with default confidence
.\.venv\Scripts\python.exe main.py --image data/sample_workers.jpg

# Custom confidence and explicit output path
.\.venv\Scripts\python.exe main.py --image data/sample_workers.jpg --conf 0.40 --output output/inspected.jpg
```

---

## 8. Automated Testing

Run the complete test suite across all modules (137 tests):

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

### Test Suite Breakdown

| Test File | Focus Area | Tests | Status |
|:---|:---|:---:|:---:|
| `tests/test_safeedge_integration.py` | Unified navigation, state management, resilience | 9 | **PASSED** |
| `tests/test_hazard_reporting.py` | CSV/JSON reports, statistics, state transitions | 20 | **PASSED** |
| `tests/test_hazard_history.py` | Session buffer, deduplication, history integration | 12 | **PASSED** |
| `tests/test_hazard_assessment.py` | Diagnostic reasoning & protective recommendations | 24 | **PASSED** |
| `tests/test_environmental_simulator.py` | Five-parameter simulation, bounds, auto drift | 30 | **PASSED** |
| `tests/test_video_processor.py` | Video processing, metadata, frame sampling | 8 | **PASSED** |
| `tests/test_tracker.py` | Centroid & IoU tracking across frames | 4 | **PASSED** |
| `tests/test_ppe_assessor.py` | Worker spatial compliance evaluation | 10 | **PASSED** |
| `tests/test_ppe_detector.py` | Detection filtering & bounding box association | 7 | **PASSED** |
| `tests/test_model_loader.py` | Weight verification & incompatible model rejection | 6 | **PASSED** |
| `tests/test_visualizer.py` | High-contrast visual rendering & status badges | 5 | **PASSED** |
| `tests/test_real_model_integration.py` | End-to-end inference on verified model | 2 | **PASSED** |
| **TOTAL** | **Full System Test Suite** | **137** | **100% PASSED** |

---

## 9. Environmental Simulation Limitations

> [!IMPORTANT]
> **SIMULATION MODE — NOT LIVE SENSOR DATA**  
> All environmental parameters (Temperature, Relative Humidity, Respirable Dust, Noise Level, and Hazardous Gas Index) are generated in software via bounded random walk stochastic modeling and operator manual controls.  
> - No physical IoT sensor hardware or atmospheric probes are connected.
> - Telemetry history and events reside in process memory (`st.session_state`) during the active session.
> - The system does NOT certify statutory occupational compliance (e.g., OSHA, NIOSH, ACGIH, or ISO standards).
