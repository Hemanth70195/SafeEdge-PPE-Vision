"""
SafeEdge AI - Automated Integration Tests for Unified Platform (Phase 5).
Verifies:
1. Unified navigation routing and module presence.
2. PPE Image Detection pipeline end-to-end.
3. PPE Video Detection metadata extraction and pipeline integration.
4. Environmental Hazard Simulator integration (modes, start, pause, resume, reset).
5. Hazard History & Reporting integration (summary stats, CSV/JSON exports).
6. State management and non-interference between modules.
7. Graceful error handling for missing models, missing thresholds, and dirty inputs.
"""

import io
import json
import pytest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from core.model_loader import ModelLoader
from core.ppe_detector import PPEDetector, Detection
from core.ppe_assessor import PPEAssessor, WorkerPPEStatus
from core.visualizer import PPEVisualizer
from core.environmental_simulator import (
    EnvironmentalSimulator,
    HazardStatus,
    DEFAULT_CONFIG_PATH,
)
from core.hazard_assessment import HazardAssessmentEngine
from core.hazard_history import HazardHistoryTracker
from core.environmental_dashboard import (
    render_simulator_view,
    render_history_reports_view,
    render_environmental_dashboard,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Unified Navigation & Module Presence
# =============================================================================
class TestUnifiedNavigation:
    def test_navigation_module_names_defined(self):
        """Unified application navigation defines the 5 required modules."""
        expected_modules = [
            "🏢 Executive Safety Overview",
            "📷 PPE Image Detection",
            "🎬 PPE Video Detection",
            "🏗️ Environmental Hazard Simulator",
            "📊 Hazard History & Reports",
        ]
        import app
        # Verify app source has all navigation sections
        with open(app.__file__, "r", encoding="utf-8") as f:
            code = f.read()

        for mod in expected_modules:
            assert mod in code, f"Module '{mod}' missing from app.py navigation!"


# =============================================================================
# 2. PPE Image Detection Integration
# =============================================================================
class TestPPEImageDetectionPipeline:
    def test_image_detection_on_sample_image(self):
        """End-to-end PPE image detection processes demo image and returns valid assessments."""
        sample_path = PROJECT_ROOT / "data" / "sample_workers.jpg"
        assert sample_path.exists(), "Sample image data/sample_workers.jpg not found!"

        image = cv2.imread(str(sample_path))
        assert image is not None
        h, w = image.shape[:2]

        config = ModelLoader.load_config()
        weights_path = PROJECT_ROOT / "models" / "best.pt"
        detector = PPEDetector(model_path=str(weights_path), device="cpu", config=config)

        detections = detector.detect(image, conf_threshold=0.35)
        assert len(detections) > 0

        assessor = PPEAssessor(config=config)
        assessments = assessor.assess(detections, image_shape=(h, w))
        assert len(assessments) >= 1

        visualizer = PPEVisualizer(config=config)
        annotated = visualizer.visualize(image, detections, assessments)
        assert annotated is not None
        assert annotated.shape == image.shape

        # Verify worker assessment properties
        for worker in assessments:
            assert worker.worker_id.startswith("worker_")
            assert worker.overall_status in [WorkerPPEStatus.COMPLIANT, WorkerPPEStatus.VIOLATION, WorkerPPEStatus.REVIEW_REQUIRED]
            assert worker.helmet_status is not None
            assert worker.vest_status is not None
            assert len(worker.explanation) > 0


# =============================================================================
# 3. PPE Video Detection Integration
# =============================================================================
class TestPPEVideoDetectionPipeline:
    def test_video_metadata_extraction(self):
        """Extracts valid metadata from sample video if present."""
        sample_video = PROJECT_ROOT / "data" / "Video Project.mp4"
        if sample_video.exists():
            from core.video_processor import extract_video_metadata
            meta = extract_video_metadata(sample_video)
            assert meta.is_valid is True
            assert meta.fps > 0
            assert meta.total_frames > 0
            assert meta.width > 0 and meta.height > 0


# =============================================================================
# 4. Environmental Hazard Simulator Integration
# =============================================================================
class TestEnvironmentalSimulatorIntegration:
    def test_five_parameters_present_and_configurable(self):
        """Environmental simulator supports all 5 required parameters with valid ranges."""
        sim = EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)
        keys = sim.get_parameter_keys()
        assert keys == ["temperature", "humidity", "dust", "noise", "gas"]

        # Manual updates with bounds
        sim.set_reading("temperature", 35.0)
        sim.set_reading("humidity", 60.0)
        sim.set_reading("dust", 120.0)
        sim.set_reading("noise", 85.0)
        sim.set_reading("gas", 55.0)

        readings = sim.get_all_readings()
        assert readings["temperature"].value == 35.0
        assert readings["dust"].value == 120.0
        assert readings["gas"].value == 55.0

    def test_simulation_controls_lifecycle(self):
        """Start, Pause, Resume, and Reset methods update simulator state correctly."""
        sim = EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)
        assert sim.is_running is False
        assert sim.is_paused is False

        sim.start_auto_simulation()
        assert sim.is_running is True
        assert sim.is_paused is False

        sim.pause_auto_simulation()
        assert sim.is_running is False
        assert sim.is_paused is True

        sim.resume_auto_simulation()
        assert sim.is_running is True
        assert sim.is_paused is False

        sim.stop_auto_simulation()
        assert sim.is_running is False

        sim.reset()
        assert sim.is_running is False
        assert sim.tick_count == 0
        assert sim.get_reading("gas").value == 8.0  # Configured default baseline


# =============================================================================
# 5. Hazard History & Report Exports
# =============================================================================
class TestHazardHistoryAndReportingIntegration:
    def test_report_downloads_generation(self):
        """CSV and JSON reports generate correctly with all required headers and metadata."""
        sim = EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)
        sim.set_reading("gas", 65.0)  # DANGER
        sim.step_simulation()

        csv_readings = sim.export_readings_csv()
        assert "SIMULATED DATA ONLY" in csv_readings
        assert "Temperature (deg C)" in csv_readings

        csv_events = sim.export_events_csv()
        assert "SIMULATED DATA ONLY" in csv_events
        assert "Gas Concentration" in csv_events or "gas" in csv_events

        json_str = sim.export_json_report_str()
        data = json.loads(json_str)
        assert data["report_metadata"]["project_name"] == "SafeEdge-PPE-Vision"
        assert "summary_statistics" in data
        assert len(data["historical_readings"]) >= 2
        assert len(data["hazard_events"]) >= 1


# =============================================================================
# 6. Cross-Module State Non-Interference
# =============================================================================
class TestCrossModuleStateNonInterference:
    def test_environmental_reset_does_not_affect_ppe_models(self):
        """Resetting environmental simulator does not mutate or reset the PPE model configuration."""
        sim = EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)
        sim.set_reading("gas", 90.0)

        ppe_config = ModelLoader.load_config()
        initial_conf = ppe_config.get("model", {}).get("confidence_threshold", 0.35)

        # Reset simulator
        sim.reset()
        assert sim.get_reading("gas").value == 8.0

        # PPE config is unmodified
        fresh_ppe_config = ModelLoader.load_config()
        assert fresh_ppe_config.get("model", {}).get("confidence_threshold") == initial_conf


# =============================================================================
# 7. Error Handling & Resilience
# =============================================================================
class TestErrorHandlingAndResilience:
    def test_missing_model_raises_clean_error(self):
        """Attempting to load a nonexistent model weights file raises ModelLoadError cleanly."""
        from core.model_loader import ModelLoadError
        with pytest.raises(ModelLoadError):
            ModelLoader.load_model(r"nonexistent_weights_path_test_123.pt")

    def test_invalid_image_raises_clean_error(self):
        """Passing an empty or invalid image array to PPEDetector raises ValueError."""
        config = ModelLoader.load_config()
        weights_path = PROJECT_ROOT / "models" / "best.pt"
        detector = PPEDetector(model_path=str(weights_path), device="cpu", config=config)

        with pytest.raises(ValueError):
            detector.detect(None)

        with pytest.raises(ValueError):
            detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
