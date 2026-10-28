# SafeEdge-PPE-Vision: Pretrained Detection Models

This directory stores YOLO model weights used for Personal Protective Equipment (PPE) detection.

> **Important**: Model weight files (`*.pt`, `*.onnx`) are excluded from Git version control via `.gitignore` to prevent bloat.

---

## 1. Primary Verified PPE Model

- **Model Identifier:** `Hansung-Cho/yolov8-ppe-detection`
- **File Name:** `best.pt`
- **Source Repository:** [Hugging Face: Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection)
- **Author:** Hansung Cho (GitHub: [Hansung-Cho/YOLO_PPE](https://github.com/Hansung-Cho/YOLO_PPE))
- **Base Architecture:** YOLOv8n (Ultralytics)
- **Framework Compatibility:** PyTorch >= 2.0.0, Ultralytics >= 8.1.0
- **License:** MIT License
- **File Size:** ~6.25 MB (6,250,090 bytes)
- **Direct Download URL:** `https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt`

### Documented Class Mapping (10 Classes)

| Class Index | Class Label | Type | PPE Compliance Role |
|:---:|:---|:---|:---|
| 0 | `Hardhat` | PPE Item | Safety Helmet / Hard Hat (Required) |
| 1 | `Mask` | PPE Item | Protective Face Mask |
| 2 | `NO-Hardhat` | Negative PPE | Missing Hard Hat Violation Indicator |
| 3 | `NO-Mask` | Negative PPE | Missing Mask Indicator |
| 4 | `NO-Safety Vest` | Negative PPE | Missing Safety Vest Violation Indicator |
| 5 | `Person` | Worker Entity | Person / Worker Bounding Box (Required) |
| 6 | `Safety Cone` | Site Asset | Safety Hazard / Cone Marker |
| 7 | `Safety Vest` | PPE Item | High-Visibility Safety Vest (Required) |
| 8 | `machinery` | Site Asset | Construction Machinery |
| 9 | `vehicle` | Site Asset | Construction Vehicle |

---

## 2. Model Verification & Compliance Criteria

A standard YOLO model (e.g., standard `yolov8n.pt` trained only on MS-COCO) detects generic objects (80 classes including person, backpack, umbrella) and **cannot** detect safety helmets or high-visibility vests. Substituting a standard COCO model is strictly prohibited.

The SafeEdge PPE Vision `ModelLoader` validates the model upon loading:
1. Verifies that the model weights exist on disk.
2. Checks that the loaded model exposes the required target classes:
   - Worker / Person entity
   - Safety Helmet / Hard Hat
   - High-Visibility Safety Vest
3. Supports flexible class synonyms mapped through `config/settings.json`.

---

## 3. How to Obtain the Model Weights

### Option A: Automatic Download (Recommended)
Run the project's model downloader utility:
```powershell
python models/download_model.py
```
This downloads `best.pt` directly into `models/best.pt` and verifies file integrity.

### Option B: Manual Download
Download `best.pt` directly from:
`https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt`
and place the file at:
`SafeEdge-PPE-Vision/models/best.pt`
