"""
SafeEdge PPE Vision - Automated Unit & Integration Tests for Environmental Hazard Reporting.
Phase 4: Historical readings, state-transition tracking, summary statistics, and CSV/JSON report exports.

Covers:
1. Historical reading creation with full metadata.
2. Correct timestamps and recorded values.
3. No duplicate records on repeated UI refreshes.
4. Hazard state-transition events (escalation and recovery: GREEN->ORANGE, ORANGE->RED, GREEN->RED, RED->ORANGE, RED->GREEN, ORANGE->GREEN).
5. Simultaneous hazards across distinct parameters.
6. Chronological ordering.
7. Summary statistics (min, max, mean, event counts, time range).
8. Empty history handling.
9. CSV export correctness (readings and events).
10. JSON export validity and schema structure.
11. Missing and invalid values handling.
12. Simulator integration and convenience methods.
"""

import io
import json
import math
import pytest
from datetime import datetime, timedelta
import pandas as pd

from core.environmental_simulator import (
    EnvironmentalSimulator,
    HazardStatus,
    DEFAULT_CONFIG_PATH,
)
from core.hazard_assessment import (
    HazardAssessmentEngine,
    SiteHazardAssessment,
)
from core.hazard_history import (
    HazardEvent,
    HazardHistoryTracker,
    SIMULATION_DISCLAIMER_TEXT,
)


@pytest.fixture
def simulator():
    """Provides a fresh EnvironmentalSimulator with integrated history."""
    return EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)


@pytest.fixture
def tracker():
    """Provides a standalone HazardHistoryTracker."""
    return HazardHistoryTracker(max_snapshots=50, max_events=50)


# =============================================================================
# 1. Historical Reading Creation & Value Integrity
# =============================================================================
class TestHistoricalReadingCreation:
    def test_historical_reading_creation_all_parameters(self, tracker):
        """Historical snapshot records timestamp, all five values, statuses, overall status, and mode."""
        readings = {
            "temperature": 25.5,
            "humidity": 50.0,
            "dust": 30.0,
            "noise": 65.0,
            "gas": 8.0,
        }
        res = tracker.record_snapshot(readings, tick=1, mode="manual")
        assert res is True
        assert tracker.snapshot_count == 1

        df = tracker.get_snapshots_dataframe()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["temperature"] == 25.5
        assert row["humidity"] == 50.0
        assert row["dust"] == 30.0
        assert row["noise"] == 65.0
        assert row["gas"] == 8.0
        assert row["Mode"] == "manual"
        assert row["Tick"] == 1
        assert "Timestamp" in row
        assert "Full_Timestamp" in row
        assert "overall_status" in row

    def test_correct_timestamps_and_iso_format(self, tracker):
        """Timestamps contain time-of-day and valid ISO 8601 formatting."""
        custom_dt = datetime(2026, 10, 28, 14, 30, 45)
        tracker.record_snapshot({"temperature": 22.0}, tick=1, timestamp=custom_dt)

        df = tracker.get_snapshots_dataframe()
        assert df["Timestamp"].iloc[0] == "14:30:45"
        assert "2026-10-28 14:30:45" in df["Full_Timestamp"].iloc[0]
        assert df["ISO_Timestamp"].iloc[0].startswith("2026-10-28T14:30:45")

    def test_individual_and_overall_statuses_recorded(self, tracker, simulator):
        """Snapshot captures individual parameter statuses and overall site status from SiteHazardAssessment."""
        readings = {
            "temperature": 38.0,  # WARNING
            "humidity": 45.0,     # SAFE
            "dust": 25.0,         # SAFE
            "noise": 70.0,        # SAFE
            "gas": 75.0,          # DANGER
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, simulator.parameters_meta)
        tracker.record_snapshot(readings, tick=2, site_assessment=site_ass, mode="auto")

        df = tracker.get_snapshots_dataframe()
        row = df.iloc[0]
        assert row["status_temperature"] == "WARNING"
        assert row["status_gas"] == "DANGER"
        assert row["status_humidity"] == "SAFE"
        assert row["overall_status"] == "DANGER"
        assert row["overall_status_color"] == "RED"
        assert row["Mode"] == "auto"


# =============================================================================
# 2. Duplicate Prevention on Repeated UI Refreshes
# =============================================================================
class TestDuplicatePrevention:
    def test_no_duplicate_records_on_repeated_ui_refreshes(self, tracker):
        """Identical readings and tick on static UI reruns do not insert duplicate rows."""
        readings = {"temperature": 23.0, "humidity": 45.0, "dust": 20.0, "noise": 65.0, "gas": 5.0}

        # First capture
        res1 = tracker.record_snapshot(readings, tick=1, mode="manual")
        assert res1 is True
        assert tracker.snapshot_count == 1

        # Simulate 10 static Streamlit page reruns
        for _ in range(10):
            res_dup = tracker.record_snapshot(readings, tick=1, mode="manual")
            assert res_dup is False  # Rejected duplicate!

        assert tracker.snapshot_count == 1

    def test_records_when_readings_change_on_same_tick(self, tracker):
        """Changing a slider value on the same tick is captured as a distinct update."""
        tracker.record_snapshot({"temperature": 23.0, "dust": 20.0}, tick=1)
        # Slider moved to 35.0
        res = tracker.record_snapshot({"temperature": 35.0, "dust": 20.0}, tick=1)
        assert res is True
        assert tracker.snapshot_count == 2


# =============================================================================
# 3. State-Transition Hazard Events
# =============================================================================
class TestStateTransitionEvents:
    def test_green_to_orange_transition(self, tracker, simulator):
        """GREEN -> ORANGE (SAFE -> WARNING) logs a WARNING escalation event."""
        # Baseline SAFE
        r0 = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 5.0}
        site0 = HazardAssessmentEngine.assess_site(r0, simulator.parameters_meta)
        tracker.record_events_from_assessment(site0, tick=0)
        assert tracker.event_count == 0

        # Dust rises to WARNING (85 µg/m³)
        r1 = {"temperature": 24.0, "humidity": 45.0, "dust": 85.0, "noise": 65.0, "gas": 5.0}
        site1 = HazardAssessmentEngine.assess_site(r1, simulator.parameters_meta)
        ev1 = tracker.record_events_from_assessment(site1, tick=1)

        assert len(ev1) == 1
        event = ev1[0]
        assert event.parameter_key == "dust"
        assert event.previous_status == HazardStatus.SAFE
        assert event.new_status == HazardStatus.WARNING
        assert event.is_warning is True
        assert event.value == 85.0

    def test_orange_to_red_transition(self, tracker, simulator):
        """ORANGE -> RED (WARNING -> DANGER) logs a DANGER escalation event."""
        # 1. Noise in WARNING (78 dB)
        r1 = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 78.0, "gas": 5.0}
        site1 = HazardAssessmentEngine.assess_site(r1, simulator.parameters_meta)
        tracker.record_events_from_assessment(site1, tick=1)

        # 2. Noise escalates to DANGER (90 dB)
        r2 = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 90.0, "gas": 5.0}
        site2 = HazardAssessmentEngine.assess_site(r2, simulator.parameters_meta)
        ev2 = tracker.record_events_from_assessment(site2, tick=2)

        assert len(ev2) == 1
        event = ev2[0]
        assert event.previous_status == HazardStatus.WARNING
        assert event.new_status == HazardStatus.DANGER
        assert event.is_danger is True

    def test_green_directly_to_red_transition(self, tracker, simulator):
        """GREEN -> RED (SAFE -> DANGER directly) logs a DANGER event."""
        # Gas spikes directly from safe baseline to critical (70 ppm)
        r = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 70.0}
        site = HazardAssessmentEngine.assess_site(r, simulator.parameters_meta)
        ev = tracker.record_events_from_assessment(site, tick=1)

        assert len(ev) == 1
        event = ev[0]
        assert event.parameter_key == "gas"
        assert event.previous_status == HazardStatus.SAFE
        assert event.new_status == HazardStatus.DANGER
        assert event.is_danger is True

    def test_red_to_orange_and_orange_to_green_recovery(self, tracker, simulator):
        """RED -> ORANGE (de-escalation) and ORANGE -> GREEN (recovery) are logged."""
        # Step 1: Start at DANGER for Gas (65 ppm)
        r_danger = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 65.0}
        site_d = HazardAssessmentEngine.assess_site(r_danger, simulator.parameters_meta)
        tracker.record_events_from_assessment(site_d, tick=1)

        # Step 2: Drops to WARNING (30 ppm)
        r_warn = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 30.0}
        site_w = HazardAssessmentEngine.assess_site(r_warn, simulator.parameters_meta)
        ev_w = tracker.record_events_from_assessment(site_w, tick=2)
        assert len(ev_w) == 1
        assert ev_w[0].previous_status == HazardStatus.DANGER
        assert ev_w[0].new_status == HazardStatus.WARNING

        # Step 3: Recovers to SAFE (8 ppm)
        r_safe = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 8.0}
        site_s = HazardAssessmentEngine.assess_site(r_safe, simulator.parameters_meta)
        ev_s = tracker.record_events_from_assessment(site_s, tick=3)
        assert len(ev_s) == 1
        assert ev_s[0].previous_status == HazardStatus.WARNING
        assert ev_s[0].new_status == HazardStatus.SAFE
        assert ev_s[0].is_recovery is True

    def test_red_directly_to_green_recovery(self, tracker, simulator):
        """RED -> GREEN (DANGER directly to SAFE) logs a recovery event."""
        # 1. Temperature DANGER (45°C)
        r1 = {"temperature": 45.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 5.0}
        site1 = HazardAssessmentEngine.assess_site(r1, simulator.parameters_meta)
        tracker.record_events_from_assessment(site1, tick=1)

        # 2. Reset/cooled directly to SAFE (22°C)
        r2 = {"temperature": 22.0, "humidity": 45.0, "dust": 25.0, "noise": 65.0, "gas": 5.0}
        site2 = HazardAssessmentEngine.assess_site(r2, simulator.parameters_meta)
        ev2 = tracker.record_events_from_assessment(site2, tick=2)
        assert len(ev2) == 1
        assert ev2[0].previous_status == HazardStatus.DANGER
        assert ev2[0].new_status == HazardStatus.SAFE
        assert ev2[0].is_recovery is True


# =============================================================================
# 4. Simultaneous Hazards & Chronological Ordering
# =============================================================================
class TestSimultaneousHazardsAndOrdering:
    def test_simultaneous_hazards_tracked_independently(self, tracker, simulator):
        """Simultaneous hazards across different parameters are both captured as distinct events."""
        # Dust enters WARNING and Gas enters DANGER on same tick
        r = {"temperature": 24.0, "humidity": 45.0, "dust": 90.0, "noise": 65.0, "gas": 75.0}
        site = HazardAssessmentEngine.assess_site(r, simulator.parameters_meta)
        new_evs = tracker.record_events_from_assessment(site, tick=5)

        assert len(new_evs) == 2
        keys = {e.parameter_key for e in new_evs}
        assert keys == {"dust", "gas"}

    def test_chronological_ordering_accessors(self, tracker, simulator):
        """Chronological events accessor provides non-decreasing tick sequence."""
        # Tick 1: Noise WARNING
        r1 = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 78.0, "gas": 5.0}
        tracker.record_events_from_assessment(HazardAssessmentEngine.assess_site(r1, simulator.parameters_meta), tick=1)

        # Tick 3: Gas DANGER
        r2 = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 78.0, "gas": 80.0}
        tracker.record_events_from_assessment(HazardAssessmentEngine.assess_site(r2, simulator.parameters_meta), tick=3)

        # Tick 5: Dust WARNING
        r3 = {"temperature": 24.0, "humidity": 45.0, "dust": 100.0, "noise": 78.0, "gas": 80.0}
        tracker.record_events_from_assessment(HazardAssessmentEngine.assess_site(r3, simulator.parameters_meta), tick=5)

        chrono = tracker.chronological_events
        assert len(chrono) == 3
        assert [e.tick for e in chrono] == [1, 3, 5]


# =============================================================================
# 5. Summary Statistics
# =============================================================================
class TestSummaryStatistics:
    def test_summary_statistics_calculations(self, tracker):
        """Summary statistics accurately compute min, max, mean, event counts, and duration."""
        # Record known snapshots
        t1 = datetime(2026, 10, 28, 10, 0, 0)
        t2 = datetime(2026, 10, 28, 10, 5, 0)
        tracker.record_snapshot({"temperature": 20.0, "dust": 30.0}, tick=1, timestamp=t1, force=True)
        tracker.record_snapshot({"temperature": 30.0, "dust": 50.0}, tick=2, timestamp=t2, force=True)

        stats = tracker.get_summary_statistics()
        assert stats["has_data"] is True
        assert stats["total_snapshots"] == 2
        assert stats["parameters"]["temperature"]["min"] == 20.0
        assert stats["parameters"]["temperature"]["max"] == 30.0
        assert stats["parameters"]["temperature"]["mean"] == 25.0
        assert stats["parameters"]["dust"]["min"] == 30.0
        assert stats["parameters"]["dust"]["max"] == 50.0
        assert stats["parameters"]["dust"]["mean"] == 40.0
        assert stats["time_range"]["duration_seconds"] == 300.0
        assert stats["time_range"]["duration_formatted"] == "00:05:00"

    def test_empty_history_summary_statistics_safe(self, tracker):
        """Empty history returns zeroed statistics without ZeroDivisionError."""
        stats = tracker.get_summary_statistics()
        assert stats["has_data"] is False
        assert stats["total_snapshots"] == 0
        assert stats["total_events"] == 0
        assert stats["parameters"]["temperature"]["mean"] == 0.0
        assert "SIMULATED DATA" in stats["disclaimer"]


# =============================================================================
# 6. CSV & JSON Report Export Correctness
# =============================================================================
class TestReportExports:
    def test_export_readings_csv_headers_and_parsing(self, tracker):
        """Exported readings CSV contains correct headers with units and can be parsed by pandas."""
        tracker.record_snapshot({
            "temperature": 24.5,
            "humidity": 50.0,
            "dust": 35.0,
            "noise": 68.0,
            "gas": 8.0,
        }, tick=1)

        csv_str = tracker.export_readings_csv()
        assert "SIMULATED DATA ONLY" in csv_str
        assert "Temperature (deg C)" in csv_str
        assert "Relative Humidity (%)" in csv_str
        assert "Dust Concentration (ug/m3)" in csv_str
        assert "Noise Level (dB)" in csv_str
        assert "Gas Concentration (ppm)" in csv_str

        # Parse CSV with pandas (skipping comments)
        df_parsed = pd.read_csv(io.StringIO(csv_str), comment="#")
        assert len(df_parsed) == 1
        assert df_parsed["Temperature (deg C)"].iloc[0] == 24.5
        assert df_parsed["Gas Concentration (ppm)"].iloc[0] == 8.0

    def test_export_events_csv_headers_and_parsing(self, tracker, simulator):
        """Exported events CSV contains required columns and parses cleanly."""
        r = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 85.0, "gas": 5.0}
        tracker.record_events_from_assessment(HazardAssessmentEngine.assess_site(r, simulator.parameters_meta), tick=1)

        csv_str = tracker.export_events_csv()
        assert "SIMULATED DATA ONLY" in csv_str
        assert "Hazard Name" in csv_str
        assert "State Transition" in csv_str
        assert "Observed Value" in csv_str
        assert "Recommended Action" in csv_str

        df_parsed = pd.read_csv(io.StringIO(csv_str), comment="#")
        assert len(df_parsed) == 1
        assert df_parsed["Hazard Name"].iloc[0] == "Noise Level"

    def test_export_json_report_schema_and_validity(self, tracker, simulator):
        """Exported JSON report contains metadata, configured thresholds, statistics, and events."""
        tracker.record_snapshot({"temperature": 24.0}, tick=1)
        r = {"temperature": 24.0, "humidity": 45.0, "dust": 25.0, "noise": 85.0, "gas": 5.0}
        tracker.record_events_from_assessment(HazardAssessmentEngine.assess_site(r, simulator.parameters_meta), tick=1)

        json_str = tracker.export_json_report_str(configured_thresholds=simulator.parameters_meta)
        data = json.loads(json_str)

        assert "report_metadata" in data
        assert "data_limitations" in data
        assert "configured_thresholds" in data
        assert "summary_statistics" in data
        assert "historical_readings" in data
        assert "hazard_events" in data

        assert data["report_metadata"]["project_name"] == "SafeEdge-PPE-Vision"
        assert "SIMULATED DATA ONLY" in data["report_metadata"]["regulatory_disclaimer"]
        assert len(data["historical_readings"]) == 1
        assert len(data["hazard_events"]) == 1

    def test_empty_exports_do_not_fail(self, tracker):
        """Exporting empty tracker produces valid, well-formed CSV and JSON."""
        csv_r = tracker.export_readings_csv()
        assert "Temperature (deg C)" in csv_r

        csv_e = tracker.export_events_csv()
        assert "Hazard Name" in csv_e

        json_str = tracker.export_json_report_str()
        data = json.loads(json_str)
        assert data["report_metadata"]["total_readings_recorded"] == 0
        assert data["report_metadata"]["total_hazard_events"] == 0


# =============================================================================
# 7. Missing and Invalid Values
# =============================================================================
class TestMissingAndInvalidValues:
    def test_missing_and_nan_values_handled_safely(self, tracker):
        """NaN, Inf, None, and missing parameter keys do not cause crashes and fall back to 0.0."""
        dirty_readings = {
            "temperature": float("nan"),
            "humidity": None,
            "dust": float("inf"),
            "noise": -float("inf"),
            # gas omitted entirely
        }
        res = tracker.record_snapshot(dirty_readings, tick=1, force=True)
        assert res is True

        df = tracker.get_snapshots_dataframe()
        row = df.iloc[0]
        assert row["temperature"] == 0.0
        assert row["humidity"] == 0.0
        assert row["dust"] == 0.0
        assert row["noise"] == 0.0
        assert row["gas"] == 0.0


# =============================================================================
# 8. Simulator Convenience Integration
# =============================================================================
class TestSimulatorIntegrationPhase4:
    def test_simulator_export_methods(self, simulator):
        """EnvironmentalSimulator exposes Phase 4 export convenience wrappers."""
        simulator.set_reading("noise", 85.0)  # WARNING
        simulator.step_simulation()

        csv_readings = simulator.export_readings_csv()
        assert "Temperature (deg C)" in csv_readings

        csv_events = simulator.export_events_csv()
        assert "Noise Level" in csv_events

        json_report = simulator.export_json_report()
        assert json_report["report_metadata"]["total_hazard_events"] >= 1

        json_str = simulator.export_json_report_str()
        assert "SafeEdge-PPE-Vision" in json_str

        stats = simulator.get_summary_statistics()
        assert stats["has_data"] is True
        assert stats["total_snapshots"] >= 2
