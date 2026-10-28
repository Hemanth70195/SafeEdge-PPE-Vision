"""
Automated Unit Tests for PPE Object Detector and Model Loader.
Uses mock YOLO model inference to validate logic without requiring remote downloads or GPU.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from core.model_loader import ModelLoader, ModelLoadError, IncompatibleModelError
from core.ppe_detector import PPEDetector, Detection


class MockBoxItem:
    """Simulates an individual Ultralytics box element."""
    def __init__(self, xyxy: List[float], conf: float, cls_id: int):
        self._xyxy = xyxy
        self._conf = conf
        self._cls = cls_id

    @property
    def xyxy(self):
        return [self._xyxy]

    @property
    def conf(self):
        return [self._conf]

    @property
    def cls(self):
        return [self._cls]


class MockBoxes:
    """Simulates Ultralytics Results.boxes container."""
    def __init__(self, box_items: List[MockBoxItem]):
        self.box_items = box_items

    def __len__(self):
        return len(self.box_items)

    def __iter__(self):
        return iter(self.box_items)


class MockYOLOResult:
    """Simulates Ultralytics single image result."""
    def __init__(self, box_items: List[MockBoxItem]):
        self.boxes = MockBoxes(box_items)


class MockYOLOModel:
    """Mock YOLO model returning preconfigured detections."""
    def __init__(self, names: Dict[int, str], detections: List[MockBoxItem] = None):
        self.names = names
        self.detections = detections or []
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def __call__(self, image, conf=0.25, verbose=False):
        # Filter detections by confidence
        filtered = [d for d in self.detections if d._conf >= conf]
        return [MockYOLOResult(filtered)]


@pytest.fixture
def valid_ppe_classes():
    """Standard verified PPE model class mapping."""
    return {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "machinery",
        9: "vehicle",
    }


def test_detector_valid_person_helmet_vest(valid_ppe_classes):
    """Test detecting person, helmet, and safety vest with valid coordinates."""
    mock_items = [
        MockBoxItem([100.0, 100.0, 200.0, 400.0], 0.92, 5),  # Person
        MockBoxItem([120.0, 95.0, 180.0, 160.0], 0.88, 0),   # Hardhat
        MockBoxItem([110.0, 170.0, 190.0, 310.0], 0.85, 7),  # Safety Vest
    ]
    mock_model = MockYOLOModel(valid_ppe_classes, mock_items)
    detector = PPEDetector(model=mock_model, conf_threshold=0.35)

    dummy_image = np.zeros((600, 600, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image)

    assert len(detections) == 3
    categories = [d.category for d in detections]
    assert "person" in categories
    assert "helmet" in categories
    assert "vest" in categories


def test_detector_confidence_filtering(valid_ppe_classes):
    """Test that detections below the confidence threshold are filtered out."""
    mock_items = [
        MockBoxItem([100.0, 100.0, 200.0, 400.0], 0.85, 5),  # High conf person
        MockBoxItem([120.0, 95.0, 180.0, 160.0], 0.20, 0),   # Low conf helmet (0.20 < 0.35)
    ]
    mock_model = MockYOLOModel(valid_ppe_classes, mock_items)
    detector = PPEDetector(model=mock_model, conf_threshold=0.35)

    dummy_image = np.zeros((600, 600, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image)

    assert len(detections) == 1
    assert detections[0].category == "person"


def test_detector_empty_detections(valid_ppe_classes):
    """Test detector returns empty list when no objects are present in the image."""
    mock_model = MockYOLOModel(valid_ppe_classes, [])
    detector = PPEDetector(model=mock_model, conf_threshold=0.35)

    dummy_image = np.zeros((600, 600, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image)
    assert detections == []


def test_detector_invalid_image_inputs(valid_ppe_classes):
    """Test detector properly raises ValueError on corrupt/invalid image inputs."""
    mock_model = MockYOLOModel(valid_ppe_classes, [])
    detector = PPEDetector(model=mock_model)

    with pytest.raises(ValueError, match="None"):
        detector.detect(None)

    with pytest.raises(ValueError, match="numpy.ndarray"):
        detector.detect("not_an_image")

    with pytest.raises(ValueError, match="invalid shape"):
        detector.detect(np.array([]))


def test_model_loader_missing_weights():
    """Test ModelLoader raises clear ModelLoadError when weight file does not exist."""
    fake_path = "models/non_existent_weights_12345.pt"
    with pytest.raises(ModelLoadError, match="not found"):
        ModelLoader.load_model(model_path=fake_path)


def test_model_loader_rejects_person_only_coco_model():
    """
    CRITICAL TEST: Ensure ModelLoader strictly rejects a person-only COCO model
    and never silently substitutes a generic model for a PPE detector.
    """
    coco_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        16: "dog",
        17: "horse",
    }
    with pytest.raises(IncompatibleModelError, match="Missing required PPE categories"):
        ModelLoader.validate_model_classes(coco_classes)


def test_model_loader_accepts_valid_ppe_classes(valid_ppe_classes):
    """Test ModelLoader correctly parses and validates genuine PPE model classes."""
    mapping = ModelLoader.validate_model_classes(valid_ppe_classes)
    assert "person" in mapping
    assert "helmet" in mapping
    assert "vest" in mapping
    assert mapping["person"] == 5
    assert mapping["helmet"] == 0
    assert mapping["vest"] == 7
