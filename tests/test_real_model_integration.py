"""
Integration Test for Real Pretrained YOLO PPE Model.
Tests actual model weight loading, real inference, and class validation.
Skipped automatically if local model weights are absent.
"""

import pytest
import numpy as np
from pathlib import Path

from core.model_loader import ModelLoader
from core.ppe_detector import PPEDetector
from core.ppe_assessor import PPEAssessor
from core.visualizer import PPEVisualizer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Real model weights (models/best.pt) not downloaded.")
def test_real_model_loading_and_classes():
    """Validates that real best.pt loads and provides genuine PPE classes."""
    model, mapping = ModelLoader.load_model(str(MODEL_PATH), force_reload=True)

    assert model is not None
    assert "person" in mapping
    assert "helmet" in mapping
    assert "vest" in mapping

    # Verify model class names in real weights
    names = model.names
    assert 0 in names and names[0].lower() in ["hardhat", "helmet"]
    assert 5 in names and names[5].lower() in ["person", "worker"]
    assert 7 in names and names[7].lower() in ["safety vest", "vest"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Real model weights (models/best.pt) not downloaded.")
def test_real_model_inference_pipeline():
    """Runs the full real detector, assessor, and visualizer pipeline."""
    detector = PPEDetector(model_path=str(MODEL_PATH), conf_threshold=0.35, device="cpu")
    assessor = PPEAssessor()
    visualizer = PPEVisualizer()

    # Synthetic image for inference smoke test
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Inference smoke test (should execute without error and return list)
    detections = detector.detect(test_img)
    assert isinstance(detections, list)

    assessments = assessor.assess(detections, image_shape=(480, 640))
    assert isinstance(assessments, list)

    annotated = visualizer.visualize(test_img, detections, assessments)
    assert annotated.shape == (480, 640, 3)
