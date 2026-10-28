"""
SafeEdge PPE Vision - Automated Unit & Integration Tests for Environmental Hazard Simulator.
Phase 1 verification suite covering:
1. Correct initialization of all five parameters and default values.
2. Valid manual reading updates and state retention.
3. Safe, warning, and danger classifications across all threshold types.
4. Correct overall site status when multiple hazards exist simultaneously.
5. Handling of invalid inputs, bounds clamping, and missing/corrupted configs.
6. Automatic simulation drift staying strictly within configured physical limits.
7. Complete reset to documented default baseline state.
8. Disclaimer presence on all reading snapshots.
"""

import math
import pytest
from pathlib import Path

from core.environmental_simulator import (
    EnvironmentalSimulator,
    EnvironmentalReading,
    HazardStatus,
    EnvironmentalConfigError,
    SIMULATED_DATA_DISCLAIMER_SHORT,
    BUILTIN_DEFAULT_CONFIG,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "environmental_thresholds.json"


@pytest.fixture
def simulator():
    """Provides a fresh EnvironmentalSimulator instance using the project config."""
    return EnvironmentalSimulator(config_path=CONFIG_PATH)


# =============================================================================
# 1. Initialization Tests
# =============================================================================
class TestSimulatorInitialization:
    def test_initialization_parameters_present(self, simulator):
        """All 5 mandatory parameters must be initialized."""
        keys = simulator.get_parameter_keys()
        expected = ["temperature", "humidity", "dust", "noise", "gas"]
        assert sorted(keys) == sorted(expected)

    def test_initial_values_match_documented_defaults(self, simulator):
        """Initial parameter values must match the documented defaults in thresholds.json."""
        readings = simulator.get_all_readings()
        assert readings["temperature"].value == 24.0
        assert readings["humidity"].value == 45.0
        assert readings["dust"].value == 28.0
        assert readings["noise"].value == 65.0
        assert readings["gas"].value == 8.0

    def test_initial_statuses_are_all_safe(self, simulator):
        """Initial baseline conditions must all evaluate to SAFE (GREEN)."""
        readings = simulator.get_all_readings()
        for key, reading in readings.items():
            assert reading.status == HazardStatus.SAFE, f"{key} should be SAFE initially"
            assert reading.status.color_label == "GREEN"
            assert reading.is_safe is True
            assert reading.is_warning is False
            assert reading.is_danger is False

    def test_initial_units_and_descriptions(self, simulator):
        """Units must correctly reflect environmental engineering standards."""
        readings = simulator.get_all_readings()
        assert readings["temperature"].unit == "°C"
        assert readings["humidity"].unit == "%"
        assert readings["dust"].unit == "µg/m³"
        assert readings["noise"].unit == "dB"
        assert readings["gas"].unit == "ppm"

    def test_disclaimer_presence(self, simulator):
        """Readings must be clearly labelled with the simulation disclaimer."""
        readings = simulator.get_all_readings()
        for key, r in readings.items():
            d = r.to_dict()
            assert "disclaimer" in d
            assert d["disclaimer"] == SIMULATED_DATA_DISCLAIMER_SHORT
            assert "SIMULATED DATA" in d["disclaimer"]
            assert "NOT LIVE SENSOR READINGS" in d["disclaimer"]
        assert "SIMULATED DATA" in simulator.disclaimer


# =============================================================================
# 2. Manual Updates & State Independence Tests
# =============================================================================
class TestManualReadingUpdates:
    def test_update_single_parameter(self, simulator):
        """Manual update of one parameter must immediately update value and status."""
        reading = simulator.set_reading("temperature", 35.0)
        assert reading.value == 35.0
        assert reading.status == HazardStatus.WARNING
        assert simulator.get_reading("temperature").value == 35.0

    def test_update_preserves_other_parameters(self, simulator):
        """Updating one slider must not reset or affect any other readings."""
        simulator.set_reading("temperature", 32.0)
        simulator.set_reading("dust", 120.0)

        # Update noise
        simulator.set_reading("noise", 80.0)

        assert simulator.get_reading("temperature").value == 32.0
        assert simulator.get_reading("dust").value == 120.0
        assert simulator.get_reading("noise").value == 80.0
        assert simulator.get_reading("humidity").value == 45.0  # untouched default
        assert simulator.get_reading("gas").value == 8.0  # untouched default

    def test_set_all_readings(self, simulator):
        """Batch setting updates all values correctly."""
        new_values = {
            "temperature": 28.0,
            "humidity": 50.0,
            "dust": 40.0,
            "noise": 68.0,
            "gas": 15.0,
        }
        res = simulator.set_all_readings(new_values)
        for k, v in new_values.items():
            assert res[k].value == v


# =============================================================================
# 3. Parameter Safety Classifications (Safe, Warning, Danger)
# =============================================================================
class TestThresholdEvaluations:
    def test_temperature_range_thresholds(self, simulator):
        """Temperature: Safe [15, 30], Warning [5, 15) and (30, 38], Danger < 5 or > 38."""
        # Safe
        st_safe, _ = simulator.evaluate_parameter_status("temperature", 24.0)
        assert st_safe == HazardStatus.SAFE
        st_safe_edge, _ = simulator.evaluate_parameter_status("temperature", 30.0)
        assert st_safe_edge == HazardStatus.SAFE

        # Warning
        st_warn_high, _ = simulator.evaluate_parameter_status("temperature", 35.0)
        assert st_warn_high == HazardStatus.WARNING
        st_warn_low, _ = simulator.evaluate_parameter_status("temperature", 10.0)
        assert st_warn_low == HazardStatus.WARNING

        # Danger
        st_dang_high, _ = simulator.evaluate_parameter_status("temperature", 42.0)
        assert st_dang_high == HazardStatus.DANGER
        st_dang_low, _ = simulator.evaluate_parameter_status("temperature", -5.0)
        assert st_dang_low == HazardStatus.DANGER

    def test_humidity_range_thresholds(self, simulator):
        """Humidity: Safe [30, 60], Warning [20, 30) and (60, 80], Danger < 20 or > 80."""
        # Safe
        assert simulator.evaluate_parameter_status("humidity", 45.0)[0] == HazardStatus.SAFE
        # Warning
        assert simulator.evaluate_parameter_status("humidity", 25.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("humidity", 70.0)[0] == HazardStatus.WARNING
        # Danger
        assert simulator.evaluate_parameter_status("humidity", 15.0)[0] == HazardStatus.DANGER
        assert simulator.evaluate_parameter_status("humidity", 85.0)[0] == HazardStatus.DANGER

    def test_dust_upper_thresholds(self, simulator):
        """Dust (PM10): Safe <= 50, Warning (50, 150], Danger > 150."""
        assert simulator.evaluate_parameter_status("dust", 20.0)[0] == HazardStatus.SAFE
        assert simulator.evaluate_parameter_status("dust", 50.0)[0] == HazardStatus.SAFE
        assert simulator.evaluate_parameter_status("dust", 95.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("dust", 150.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("dust", 210.0)[0] == HazardStatus.DANGER

    def test_noise_upper_thresholds(self, simulator):
        """Noise: Safe <= 70 dB, Warning (70, 85] dB, Danger > 85 dB."""
        assert simulator.evaluate_parameter_status("noise", 60.0)[0] == HazardStatus.SAFE
        assert simulator.evaluate_parameter_status("noise", 70.0)[0] == HazardStatus.SAFE
        assert simulator.evaluate_parameter_status("noise", 78.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("noise", 85.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("noise", 95.0)[0] == HazardStatus.DANGER

    def test_gas_upper_thresholds(self, simulator):
        """Gas: Safe <= 25 ppm, Warning (25, 50] ppm, Danger > 50 ppm."""
        assert simulator.evaluate_parameter_status("gas", 5.0)[0] == HazardStatus.SAFE
        assert simulator.evaluate_parameter_status("gas", 25.0)[0] == HazardStatus.SAFE
        assert simulator.evaluate_parameter_status("gas", 35.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("gas", 50.0)[0] == HazardStatus.WARNING
        assert simulator.evaluate_parameter_status("gas", 75.0)[0] == HazardStatus.DANGER


# =============================================================================
# 4. Overall Site Hazard Status Aggregation Tests
# =============================================================================
class TestOverallSiteHazardStatus:
    def test_overall_status_all_safe(self, simulator):
        """When all readings are SAFE, overall status must be SAFE (GREEN)."""
        status, msg, hazards = simulator.get_overall_site_status()
        assert status == HazardStatus.SAFE
        assert status.color_label == "GREEN"
        assert len(hazards) == 0

    def test_overall_status_single_warning(self, simulator):
        """One WARNING promotes overall status to WARNING (ORANGE)."""
        simulator.set_reading("noise", 75.0)  # Warning
        status, msg, hazards = simulator.get_overall_site_status()
        assert status == HazardStatus.WARNING
        assert status.color_label == "ORANGE"
        assert "Noise Level" in hazards

    def test_overall_status_multiple_warnings(self, simulator):
        """Multiple WARNINGs result in overall WARNING listing all active warning hazards."""
        simulator.set_reading("noise", 75.0)  # Warning
        simulator.set_reading("dust", 110.0)  # Warning
        status, msg, hazards = simulator.get_overall_site_status()
        assert status == HazardStatus.WARNING
        assert "Noise Level" in hazards
        assert "Dust Concentration" in hazards

    def test_overall_status_danger_takes_precedence(self, simulator):
        """A single DANGER must override any number of WARNINGs to yield DANGER (RED)."""
        simulator.set_reading("noise", 75.0)  # Warning
        simulator.set_reading("dust", 110.0)  # Warning
        simulator.set_reading("gas", 65.0)  # DANGER

        status, msg, hazards = simulator.get_overall_site_status()
        assert status == HazardStatus.DANGER
        assert status.color_label == "RED"
        assert "Gas Concentration" in hazards

    def test_overall_status_multiple_dangers(self, simulator):
        """Multiple DANGERs report all active danger hazards."""
        simulator.set_reading("gas", 80.0)  # DANGER
        simulator.set_reading("noise", 100.0)  # DANGER

        status, msg, hazards = simulator.get_overall_site_status()
        assert status == HazardStatus.DANGER
        assert "Gas Concentration" in hazards
        assert "Noise Level" in hazards


# =============================================================================
# 5. Robustness, Clamping, and Missing Config Handling
# =============================================================================
class TestRobustnessAndErrorHandling:
    def test_invalid_parameter_key(self, simulator):
        """Querying or updating an unconfigured parameter key raises KeyError."""
        with pytest.raises(KeyError):
            simulator.set_reading("radiation", 1.0)
        with pytest.raises(KeyError):
            simulator.get_reading("radiation")
        with pytest.raises(KeyError):
            simulator.evaluate_parameter_status("radiation", 1.0)

    def test_invalid_numerical_values(self, simulator):
        """Non-numeric or non-finite inputs raise ValueError."""
        with pytest.raises(ValueError):
            simulator.set_reading("temperature", "hot")  # type: ignore
        with pytest.raises(ValueError):
            simulator.set_reading("temperature", float("nan"))
        with pytest.raises(ValueError):
            simulator.set_reading("temperature", float("inf"))

    def test_bounds_clamping(self, simulator):
        """Values exceeding min/max range limits are clamped to boundaries."""
        # Temperature min is -10.0, max is 55.0
        r_low = simulator.set_reading("temperature", -50.0)
        assert r_low.value == -10.0

        r_high = simulator.set_reading("temperature", 100.0)
        assert r_high.value == 55.0

    def test_missing_config_raises_when_fallback_disabled(self):
        """Non-existent config file raises EnvironmentalConfigError if fallback disabled."""
        with pytest.raises(EnvironmentalConfigError):
            EnvironmentalSimulator(
                config_path=Path("non_existent_config.json"),
                allow_fallback_defaults=False,
            )

    def test_missing_config_falls_back_gracefully(self):
        """Non-existent config falls back to builtin defaults if allowed."""
        sim = EnvironmentalSimulator(
            config_path=Path("non_existent_config.json"),
            allow_fallback_defaults=True,
        )
        assert sim.get_reading("temperature").value == 24.0

    def test_corrupted_config_raises_error(self, tmp_path):
        """Malformed JSON config raises EnvironmentalConfigError."""
        corrupt_file = tmp_path / "corrupt_config.json"
        corrupt_file.write_text("{ incomplete json ...", encoding="utf-8")
        with pytest.raises(EnvironmentalConfigError):
            EnvironmentalSimulator(config_path=corrupt_file, allow_fallback_defaults=False)


# =============================================================================
# 6. Automatic Simulation Mode Tests
# =============================================================================
class TestAutomaticSimulation:
    def test_start_pause_resume_state(self, simulator):
        """State flags must accurately track simulation controls."""
        assert simulator.is_running is False
        assert simulator.is_paused is False

        simulator.start_auto_simulation()
        assert simulator.is_running is True
        assert simulator.is_paused is False

        simulator.pause_auto_simulation()
        assert simulator.is_running is False
        assert simulator.is_paused is True

        simulator.resume_auto_simulation()
        assert simulator.is_running is True
        assert simulator.is_paused is False

    def test_step_simulation_increments_tick(self, simulator):
        """Stepping simulation increments tick count and produces readings."""
        assert simulator.tick_count == 0
        simulator.step_simulation()
        assert simulator.tick_count == 1
        simulator.step_simulation()
        assert simulator.tick_count == 2

    def test_simulation_values_remain_within_configured_ranges(self, simulator):
        """Values generated across 100 automatic simulation steps must strictly obey min/max limits."""
        meta = simulator.parameters_meta
        for _ in range(100):
            readings = simulator.step_simulation(drift_scale=1.5)
            for k, r in readings.items():
                min_val = meta[k]["min_value"]
                max_val = meta[k]["max_value"]
                assert min_val <= r.value <= max_val, (
                    f"{k} value {r.value} drifted outside [{min_val}, {max_val}]"
                )

    def test_simulation_state_snapshot(self, simulator):
        """Simulation snapshot contains all operational metadata."""
        state = simulator.get_simulation_state()
        assert "is_running" in state
        assert "is_paused" in state
        assert "overall_status" in state
        assert "readings" in state
        assert len(state["readings"]) == 5

    def test_simulation_works_without_internet_or_keys(self, simulator):
        """Simulation operates completely self-contained with zero API keys or external services."""
        # Step simulation several times
        for _ in range(5):
            readings = simulator.step_simulation()
            assert len(readings) == 5
            for k in ["temperature", "humidity", "dust", "noise", "gas"]:
                assert k in readings
                assert isinstance(readings[k].value, float)


# =============================================================================
# 7. Reset Functionality Tests
# =============================================================================
class TestResetFunctionality:
    def test_reset_restores_defaults(self, simulator):
        """Resetting must restore initial documented values, clear running status, and reset tick."""
        simulator.start_auto_simulation()
        simulator.set_reading("gas", 90.0)  # DANGER
        simulator.set_reading("noise", 110.0)  # DANGER
        simulator.step_simulation()

        assert simulator.get_reading("gas").status == HazardStatus.DANGER
        assert simulator.tick_count > 0

        # Reset
        res = simulator.reset()

        assert res["gas"].value == 8.0
        assert res["noise"].value == 65.0
        assert res["temperature"].value == 24.0
        assert res["humidity"].value == 45.0
        assert res["dust"].value == 28.0

        assert simulator.is_running is False
        assert simulator.is_paused is False
        assert simulator.tick_count == 0
        assert simulator.get_overall_site_status()[0] == HazardStatus.SAFE
