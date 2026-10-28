"""
SafeEdge PPE Vision - Automated Unit & Integration Tests for Environmental Hazard Assessment Engine.
Phase 2 verification suite covering:
1. GREEN, ORANGE, and RED classifications for every hazard parameter.
2. Multiple simultaneous hazards sorted strictly by severity.
3. Correct overall site status aggregation based on the most severe active hazard.
4. Specific required recommendations for each parameter:
   * Temperature: Rest, cooling, hydration, site heat-safety procedures, cold protection.
   * Dust: Source control, suitable ventilation, respiratory-protection procedures.
   * Noise: Reduce exposure, appropriate hearing protection.
   * Gas: Emergency procedures, leaving the affected area (evacuate), calibrated gas detectors.
   * Humidity: Arid/static/respiratory concerns vs. evaporative cooling/heat index/condensation concerns.
5. Robust error handling for invalid, non-numeric, NaN, infinite, and missing readings.
6. Boundary values at all configured safe, warning, and danger thresholds.
7. End-to-end integration with EnvironmentalSimulator.
"""

import math
import pytest
from pathlib import Path

from core.environmental_simulator import (
    EnvironmentalSimulator,
    EnvironmentalReading,
    HazardStatus,
    DEFAULT_CONFIG_PATH,
)
from core.hazard_assessment import (
    HazardAssessmentEngine,
    HazardAssessment,
    SiteHazardAssessment,
)


@pytest.fixture
def simulator():
    """Provides a fresh EnvironmentalSimulator instance using project config."""
    return EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)


@pytest.fixture
def params_meta(simulator):
    """Provides parameter configuration metadata from the simulator."""
    return simulator.parameters_meta


# =============================================================================
# 1. Classification Tests (GREEN, ORANGE, RED for all 5 parameters)
# =============================================================================
class TestHazardClassifications:
    def test_temperature_classifications(self, params_meta):
        meta = params_meta["temperature"]
        # Safe (GREEN)
        ass_safe = HazardAssessmentEngine.assess_parameter("temperature", 24.0, meta)
        assert ass_safe.status == HazardStatus.SAFE
        assert ass_safe.is_safe is True

        # Warning Low (ORANGE)
        ass_warn_low = HazardAssessmentEngine.assess_parameter("temperature", 10.0, meta)
        assert ass_warn_low.status == HazardStatus.WARNING
        assert ass_warn_low.is_warning is True

        # Warning High (ORANGE)
        ass_warn_high = HazardAssessmentEngine.assess_parameter("temperature", 34.0, meta)
        assert ass_warn_high.status == HazardStatus.WARNING
        assert ass_warn_high.is_warning is True

        # Danger Low (RED)
        ass_dang_low = HazardAssessmentEngine.assess_parameter("temperature", -5.0, meta)
        assert ass_dang_low.status == HazardStatus.DANGER
        assert ass_dang_low.is_danger is True

        # Danger High (RED)
        ass_dang_high = HazardAssessmentEngine.assess_parameter("temperature", 42.0, meta)
        assert ass_dang_high.status == HazardStatus.DANGER
        assert ass_dang_high.is_danger is True

    def test_humidity_classifications(self, params_meta):
        meta = params_meta["humidity"]
        # Safe (GREEN)
        ass_safe = HazardAssessmentEngine.assess_parameter("humidity", 45.0, meta)
        assert ass_safe.status == HazardStatus.SAFE

        # Warning Low (ORANGE)
        ass_warn_low = HazardAssessmentEngine.assess_parameter("humidity", 25.0, meta)
        assert ass_warn_low.status == HazardStatus.WARNING

        # Warning High (ORANGE)
        ass_warn_high = HazardAssessmentEngine.assess_parameter("humidity", 70.0, meta)
        assert ass_warn_high.status == HazardStatus.WARNING

        # Danger Low (RED)
        ass_dang_low = HazardAssessmentEngine.assess_parameter("humidity", 15.0, meta)
        assert ass_dang_low.status == HazardStatus.DANGER

        # Danger High (RED)
        ass_dang_high = HazardAssessmentEngine.assess_parameter("humidity", 85.0, meta)
        assert ass_dang_high.status == HazardStatus.DANGER

    def test_dust_classifications(self, params_meta):
        meta = params_meta["dust"]
        # Safe (GREEN)
        assert HazardAssessmentEngine.assess_parameter("dust", 25.0, meta).status == HazardStatus.SAFE
        # Warning (ORANGE)
        assert HazardAssessmentEngine.assess_parameter("dust", 80.0, meta).status == HazardStatus.WARNING
        # Danger (RED)
        assert HazardAssessmentEngine.assess_parameter("dust", 200.0, meta).status == HazardStatus.DANGER

    def test_noise_classifications(self, params_meta):
        meta = params_meta["noise"]
        # Safe (GREEN)
        assert HazardAssessmentEngine.assess_parameter("noise", 60.0, meta).status == HazardStatus.SAFE
        # Warning (ORANGE)
        assert HazardAssessmentEngine.assess_parameter("noise", 78.0, meta).status == HazardStatus.WARNING
        # Danger (RED)
        assert HazardAssessmentEngine.assess_parameter("noise", 95.0, meta).status == HazardStatus.DANGER

    def test_gas_classifications(self, params_meta):
        meta = params_meta["gas"]
        # Safe (GREEN)
        assert HazardAssessmentEngine.assess_parameter("gas", 10.0, meta).status == HazardStatus.SAFE
        # Warning (ORANGE)
        assert HazardAssessmentEngine.assess_parameter("gas", 35.0, meta).status == HazardStatus.WARNING
        # Danger (RED)
        assert HazardAssessmentEngine.assess_parameter("gas", 65.0, meta).status == HazardStatus.DANGER


# =============================================================================
# 2. Specific Recommended Protective Actions Tests
# =============================================================================
class TestProtectiveActionRecommendations:
    def test_high_temperature_recommendations(self, params_meta):
        """For high temperature, must recommend rest, cooling, hydration, and site heat-safety procedures."""
        meta = params_meta["temperature"]
        # Warning high
        ass_w = HazardAssessmentEngine.assess_parameter("temperature", 35.0, meta)
        rec_text_w = " ".join(ass_w.recommended_actions).lower()
        assert "rest" in rec_text_w
        assert "cool" in rec_text_w
        assert "hydration" in rec_text_w or "water" in rec_text_w
        assert "heat-safety" in rec_text_w or "heat" in rec_text_w

        # Danger high
        ass_d = HazardAssessmentEngine.assess_parameter("temperature", 42.0, meta)
        rec_text_d = " ".join(ass_d.recommended_actions).lower()
        assert "rest" in rec_text_d
        assert "cool" in rec_text_d
        assert "hydration" in rec_text_d or "water" in rec_text_d
        assert "emergency" in rec_text_d or "procedures" in rec_text_d

    def test_dust_recommendations(self, params_meta):
        """For excessive dust, must recommend controlling dust source, suitable ventilation, and respiratory protection."""
        meta = params_meta["dust"]
        ass = HazardAssessmentEngine.assess_parameter("dust", 180.0, meta)
        rec_text = " ".join(ass.recommended_actions).lower()
        assert "dust source" in rec_text or "suppression" in rec_text
        assert "ventilation" in rec_text
        assert "respiratory-protection" in rec_text or "respirator" in rec_text

    def test_noise_recommendations(self, params_meta):
        """For excessive noise, must recommend reducing exposure and appropriate hearing protection."""
        meta = params_meta["noise"]
        ass = HazardAssessmentEngine.assess_parameter("noise", 92.0, meta)
        rec_text = " ".join(ass.recommended_actions).lower()
        assert "exposure" in rec_text or "rotation" in rec_text
        assert "hearing protection" in rec_text or "earmuffs" in rec_text

    def test_gas_recommendations(self, params_meta):
        """For dangerous gas readings, must recommend emergency procedures, leaving the area, and calibrated gas detectors."""
        meta = params_meta["gas"]
        ass = HazardAssessmentEngine.assess_parameter("gas", 70.0, meta)
        rec_text = " ".join(ass.recommended_actions).lower()
        assert "emergency procedures" in rec_text or "emergency" in rec_text
        assert "leave" in rec_text or "evacuation" in rec_text or "evacuate" in rec_text
        assert "calibrated" in rec_text and "detector" in rec_text

    def test_humidity_site_specific_recommendations(self, params_meta):
        """For humidity outside configured range, explain relevant site-specific concern and recommended action."""
        meta = params_meta["humidity"]
        # Low humidity: static electricity / dust suspension / dryness
        ass_low = HazardAssessmentEngine.assess_parameter("humidity", 15.0, meta)
        reason_low = ass_low.reason.lower()
        rec_low = " ".join(ass_low.recommended_actions).lower()
        assert "static" in reason_low or "electrostatic" in reason_low or "arid" in reason_low
        assert "grounding" in rec_low or "static" in rec_low or "misting" in rec_low

        # High humidity: evaporative cooling impaired / heat index / condensation / slip
        ass_high = HazardAssessmentEngine.assess_parameter("humidity", 75.0, meta)
        reason_high = ass_high.reason.lower()
        rec_high = " ".join(ass_high.recommended_actions).lower()
        assert "evaporative" in reason_high or "heat index" in reason_high or "condensation" in reason_high
        assert "ventilation" in rec_high or "rest" in rec_high or "condensation" in rec_high or "slip" in rec_high


# =============================================================================
# 3. Overall Site Status & Severity Sorting Tests
# =============================================================================
class TestSiteAssessmentAndSeverityOrdering:
    def test_all_safe_site_status(self, params_meta):
        """When all readings are SAFE, overall status must be SAFE (GREEN) with no active hazards."""
        readings = {
            "temperature": 24.0,
            "humidity": 45.0,
            "dust": 28.0,
            "noise": 65.0,
            "gas": 8.0,
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, params_meta)
        assert site_ass.overall_status == HazardStatus.SAFE
        assert site_ass.has_hazards is False
        assert len(site_ass.active_hazards) == 0
        assert len(site_ass.safe_parameters) == 5

    def test_single_warning_site_status(self, params_meta):
        """Single WARNING results in overall WARNING with that parameter active."""
        readings = {
            "temperature": 24.0,
            "humidity": 45.0,
            "dust": 100.0,  # Warning
            "noise": 65.0,
            "gas": 8.0,
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, params_meta)
        assert site_ass.overall_status == HazardStatus.WARNING
        assert site_ass.warning_count == 1
        assert site_ass.danger_count == 0
        assert len(site_ass.active_hazards) == 1
        assert site_ass.active_hazards[0].parameter_key == "dust"

    def test_danger_overrides_warning(self, params_meta):
        """DANGER overrides WARNING, setting overall status to DANGER (RED)."""
        readings = {
            "temperature": 35.0,  # Warning
            "humidity": 45.0,
            "dust": 90.0,   # Warning
            "noise": 95.0,  # DANGER
            "gas": 8.0,
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, params_meta)
        assert site_ass.overall_status == HazardStatus.DANGER
        assert site_ass.danger_count == 1
        assert site_ass.warning_count == 2

    def test_multiple_hazards_severity_ordering(self, params_meta):
        """Multiple simultaneous hazards must be sorted strictly by severity (DANGER first, then WARNING)."""
        readings = {
            "temperature": 35.0,  # WARNING (severity 1)
            "humidity": 85.0,     # DANGER (severity 2)
            "dust": 120.0,        # WARNING (severity 1)
            "noise": 100.0,       # DANGER (severity 2)
            "gas": 8.0,           # SAFE (severity 0)
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, params_meta)
        assert len(site_ass.active_hazards) == 4

        # DANGER hazards must appear before WARNING hazards
        severities = [h.severity for h in site_ass.active_hazards]
        assert severities == [2, 2, 1, 1]
        assert site_ass.active_hazards[0].status == HazardStatus.DANGER
        assert site_ass.active_hazards[1].status == HazardStatus.DANGER
        assert site_ass.active_hazards[2].status == HazardStatus.WARNING
        assert site_ass.active_hazards[3].status == HazardStatus.WARNING


# =============================================================================
# 4. Boundary Value Testing
# =============================================================================
class TestBoundaryValues:
    def test_temperature_exact_boundaries(self, params_meta):
        meta = params_meta["temperature"]
        # Exactly on safe boundaries
        assert HazardAssessmentEngine.assess_parameter("temperature", 15.0, meta).status == HazardStatus.SAFE
        assert HazardAssessmentEngine.assess_parameter("temperature", 30.0, meta).status == HazardStatus.SAFE

        # Just outside safe boundaries
        assert HazardAssessmentEngine.assess_parameter("temperature", 14.99, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("temperature", 30.01, meta).status == HazardStatus.WARNING

        # Exactly on warning boundaries
        assert HazardAssessmentEngine.assess_parameter("temperature", 5.0, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("temperature", 38.0, meta).status == HazardStatus.WARNING

        # Just past warning into danger
        assert HazardAssessmentEngine.assess_parameter("temperature", 4.99, meta).status == HazardStatus.DANGER
        assert HazardAssessmentEngine.assess_parameter("temperature", 38.01, meta).status == HazardStatus.DANGER

    def test_dust_exact_boundaries(self, params_meta):
        meta = params_meta["dust"]
        assert HazardAssessmentEngine.assess_parameter("dust", 50.0, meta).status == HazardStatus.SAFE
        assert HazardAssessmentEngine.assess_parameter("dust", 50.01, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("dust", 150.0, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("dust", 150.01, meta).status == HazardStatus.DANGER

    def test_noise_exact_boundaries(self, params_meta):
        meta = params_meta["noise"]
        assert HazardAssessmentEngine.assess_parameter("noise", 70.0, meta).status == HazardStatus.SAFE
        assert HazardAssessmentEngine.assess_parameter("noise", 70.01, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("noise", 85.0, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("noise", 85.01, meta).status == HazardStatus.DANGER

    def test_gas_exact_boundaries(self, params_meta):
        meta = params_meta["gas"]
        assert HazardAssessmentEngine.assess_parameter("gas", 25.0, meta).status == HazardStatus.SAFE
        assert HazardAssessmentEngine.assess_parameter("gas", 25.01, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("gas", 50.0, meta).status == HazardStatus.WARNING
        assert HazardAssessmentEngine.assess_parameter("gas", 50.01, meta).status == HazardStatus.DANGER


# =============================================================================
# 5. Invalid and Missing Readings Robustness
# =============================================================================
class TestInvalidAndMissingReadings:
    def test_none_value_handled_safely(self, params_meta):
        """None value handled as precautionary warning without unhandled exception."""
        ass = HazardAssessmentEngine.assess_parameter("temperature", None, params_meta["temperature"])
        assert ass.status == HazardStatus.WARNING
        assert "INVALID OR MISSING" in ass.reason

    def test_nan_and_inf_handled_safely(self, params_meta):
        """NaN and infinite values handled safely."""
        ass_nan = HazardAssessmentEngine.assess_parameter("dust", float("nan"), params_meta["dust"])
        assert ass_nan.status == HazardStatus.WARNING

        ass_inf = HazardAssessmentEngine.assess_parameter("noise", float("inf"), params_meta["noise"])
        assert ass_inf.status == HazardStatus.WARNING

    def test_missing_parameter_in_site_assessment(self, params_meta):
        """Missing parameters in readings dictionary handled safely."""
        partial_readings = {"temperature": 24.0}  # 4 parameters missing
        site_ass = HazardAssessmentEngine.assess_site(partial_readings, params_meta)
        assert len(site_ass.all_assessments) == 5
        # The missing parameters should evaluate with warning
        assert site_ass.all_assessments["gas"].status == HazardStatus.WARNING

    def test_empty_readings_dict_handled_safely(self, params_meta):
        """Empty readings dictionary evaluates safely."""
        site_ass = HazardAssessmentEngine.assess_site({}, params_meta)
        assert len(site_ass.all_assessments) == 5
        assert site_ass.overall_status == HazardStatus.WARNING


# =============================================================================
# 6. EnvironmentalSimulator Integration Tests
# =============================================================================
class TestSimulatorIntegration:
    def test_simulator_assess_current_hazards_method(self, simulator):
        """Simulator assess_current_hazards produces valid SiteHazardAssessment."""
        simulator.reset()
        site_ass = simulator.assess_current_hazards()
        assert isinstance(site_ass, SiteHazardAssessment)
        assert site_ass.overall_status == HazardStatus.SAFE

        # Trigger multi-hazard state
        simulator.set_reading("gas", 80.0)      # DANGER
        simulator.set_reading("noise", 82.0)    # WARNING
        site_ass_after = simulator.assess_current_hazards()
        assert site_ass_after.overall_status == HazardStatus.DANGER
        assert len(site_ass_after.active_hazards) == 2
        assert site_ass_after.active_hazards[0].parameter_key == "gas"
        assert site_ass_after.active_hazards[1].parameter_key == "noise"

    def test_simulator_assess_parameter_method(self, simulator):
        """Simulator assess_parameter produces valid HazardAssessment."""
        simulator.set_reading("dust", 160.0)
        ass = simulator.assess_parameter("dust")
        assert isinstance(ass, HazardAssessment)
        assert ass.status == HazardStatus.DANGER
        assert "respiratory" in " ".join(ass.recommended_actions).lower()
