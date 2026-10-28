"""
Automated Unit Tests for PPE Visualizer.
Verifies bounding box rendering, worker compliance banners, color coding, and legend generation.
"""

import pytest
import numpy as np
import cv2

from core.ppe_detector import Detection
from core.ppe_assessor import WorkerAssessment, PPEStatus, WorkerPPEStatus
from core.visualizer import PPEVisualizer


@pytest.fixture
def visualizer():
    return PPEVisualizer()


def test_visualizer_renders_annotated_image(visualizer):
    """Test visualizer produces a valid NumPy array with annotations and correct shape."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    person = Detection("Person", 0.90, [100.0, 100.0, 200.0, 400.0], "person", 5)
    helmet = Detection("Hardhat", 0.88, [120.0, 95.0, 180.0, 160.0], "helmet", 0)
    vest = Detection("Safety Vest", 0.85, [110.0, 170.0, 190.0, 310.0], "vest", 7)

    assessment = WorkerAssessment(
        worker_id="worker_1",
        person_box=person.bbox,
        person_confidence=0.90,
        helmet_status=PPEStatus.DETECTED,
        vest_status=PPEStatus.DETECTED,
        overall_status=WorkerPPEStatus.COMPLIANT,
        confidence_scores={"person": 0.90, "helmet": 0.88, "vest": 0.85},
        explanation="Fully compliant",
    )

    annotated = visualizer.visualize(img, [person, helmet, vest], [assessment])

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    # Ensure drawing occurred (annotated image is not entirely black)
    assert np.any(annotated > 0)


def test_visualizer_status_colors(visualizer):
    """Verify specific color definitions for compliance, violation, and review required."""
    compliant_color = visualizer._get_status_color(WorkerPPEStatus.COMPLIANT)
    violation_color = visualizer._get_status_color(WorkerPPEStatus.VIOLATION)
    review_color = visualizer._get_status_color(WorkerPPEStatus.REVIEW_REQUIRED)

    # Green for COMPLIANT (BGR: G channel should be dominant)
    assert compliant_color[1] > compliant_color[0] and compliant_color[1] > compliant_color[2]
    # Red for VIOLATION (BGR: R channel should be dominant)
    assert violation_color[2] > violation_color[0] and violation_color[2] > violation_color[1]
    # Orange for REVIEW_REQUIRED (BGR: (0, 140, 255) -> R and G high, B low)
    assert review_color[2] > 200 and review_color[1] > 100 and review_color[0] < 50


def test_visualizer_legend_generation(visualizer):
    """Test legend is drawn on the image canvas."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    with_legend = visualizer.draw_legend(img, x=10, y=10)
    assert np.any(with_legend > 0)


def test_visualizer_empty_scene(visualizer):
    """Test visualization handles scenes with zero detections cleanly."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    annotated = visualizer.visualize(img, [], [])
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape


def test_visualizer_invalid_image_raises_error(visualizer):
    """Test visualizer rejects None or empty images."""
    with pytest.raises(ValueError, match="Invalid image"):
        visualizer.visualize(None, [], [])

    with pytest.raises(ValueError, match="Invalid image"):
        visualizer.visualize(np.array([]), [], [])
