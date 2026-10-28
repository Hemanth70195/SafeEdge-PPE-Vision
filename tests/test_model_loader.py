"""
Unit Tests for ModelLoader.
Validates model weights discovery, caching, device resolution, and class compatibility checks.
"""

import pytest
from core.model_loader import ModelLoader, ModelLoadError, IncompatibleModelError


def test_resolve_device_cpu():
    """Verify CPU resolution."""
    assert ModelLoader.resolve_device("cpu") == "cpu"


def test_resolve_device_auto():
    """Verify auto device resolution returns either 'cpu' or 'cuda'."""
    device = ModelLoader.resolve_device("auto")
    assert device in ["cpu", "cuda"]


def test_missing_model_weights_raises_error():
    """Verify non-existent weight file raises ModelLoadError."""
    with pytest.raises(ModelLoadError, match="not found"):
        ModelLoader.load_model("non_existent_weights_xyz.pt")


def test_coco_person_only_model_is_strictly_rejected():
    """
    CRITICAL RULE TEST: A generic COCO model containing only 'person'
    must NEVER be accepted as a PPE detector.
    """
    coco_classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}
    with pytest.raises(IncompatibleModelError, match="Missing required PPE categories"):
        ModelLoader.validate_model_classes(coco_classes)


def test_empty_class_dict_rejected():
    """Verify empty class dictionary is rejected."""
    with pytest.raises(IncompatibleModelError, match="empty or invalid"):
        ModelLoader.validate_model_classes({})


def test_valid_ppe_classes_accepted():
    """Verify valid PPE class dictionary maps properly."""
    valid_classes = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Safety Vest",
        4: "Person",
        5: "Safety Vest",
    }
    mapping = ModelLoader.validate_model_classes(valid_classes)
    assert mapping["person"] == 4
    assert mapping["helmet"] == 0
    assert mapping["vest"] == 5
    assert mapping["no_helmet"] == 2
    assert mapping["no_vest"] == 3
