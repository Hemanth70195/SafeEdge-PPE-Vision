"""
SafeEdge PPE Vision - Automated Unit & Integration Tests for Hazard History & Event Logging.
Phase 3 verification suite covering:
1. Telemetry snapshot recording and rolling buffer capacity.
2. Safe DataFrame conversion and empty-state handling without errors.
3. Event logging for WARNING and DANGER hazard states.
4. Robust deduplication preventing duplicate event rows on static UI reruns.
5. End-to-end integration with EnvironmentalSimulator.
6. Reset clearing history and recording clean baseline.
"""

import pytest
from datetime import datetime
from pathlib import Path

from core.environmental_simulator import (
    EnvironmentalSimulator,
    HazardStatus,
    DEFAULT_CONFIG_PATH,
)
from core.hazard_assessment import (
    HazardAssessmentEngine,
    HazardAssessment,
    SiteHazardAssessment,
)
from core.hazard_history import (
    HazardEvent,
    HazardHistoryTracker,
)


@pytest.fixture
def simulator():
    """Provides a fresh EnvironmentalSimulator with integrated history."""
    return EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)


@pytest.fixture
def tracker():
    """Provides a standalone HazardHistoryTracker."""
    return HazardHistoryTracker(max_snapshots=10, max_events=20)


# =============================================================================
# 1. Snapshot Recording & Rolling Buffer Tests
# =============================================================================
class TestSnapshotRecording:
    def test_initial_state_empty(self, tracker):
        """Initial tracker starts with empty snapshots and events."""
        assert tracker.has_snapshots is False
        assert tracker.has_events is False
        assert tracker.snapshot_count == 0
        assert tracker.event_count == 0

    def test_record_single_snapshot(self, tracker):
        """Recording a snapshot stores all 5 parameters."""
        readings = {
            "temperature": 24.5,
            "humidity": 48.0,
            "dust": 32.0,
            "noise": 66.0,
            "gas": 9.0,
        }
        tracker.record_snapshot(readings, tick=1)
        assert tracker.has_snapshots is True
        assert tracker.snapshot_count == 1

        df = tracker.get_snapshots_dataframe()
        assert len(df) == 1
        assert df["temperature"].iloc[0] == 24.5
        assert df["dust"].iloc[0] == 32.0
        assert df["Tick"].iloc[0] == 1

    def test_rolling_buffer_capacity(self, tracker):
        """Snapshots exceeding max_snapshots drop oldest entries."""
        for i in range(15):
            tracker.record_snapshot({"temperature": float(i)}, tick=i)

        assert tracker.snapshot_count == tracker.max_snapshots  # 10
        df = tracker.get_snapshots_dataframe()
        assert len(df) == 10
        # Oldest kept should be tick 5 (15 - 10)
        assert df["Tick"].iloc[0] == 5
        assert df["Tick"].iloc[-1] == 14

    def test_empty_snapshots_dataframe_safe(self, tracker):
        """Empty snapshot DataFrame returns expected columns without raising exceptions."""
        df = tracker.get_snapshots_dataframe()
        assert df is not None
        assert len(df) == 0
        assert "Timestamp" in df.columns
        assert "temperature" in df.columns


# =============================================================================
# 2. Event Logging & Deduplication Tests
# =============================================================================
class TestEventLoggingAndDeduplication:
    def test_empty_events_dataframe_safe(self, tracker):
        """Empty event DataFrame returns expected columns without raising exceptions."""
        df = tracker.get_events_dataframe()
        assert df is not None
        assert len(df) == 0
        assert "Hazard Type" in df.columns
        assert "Severity" in df.columns

    def test_log_warning_and_danger_events(self, tracker, simulator):
        """Active WARNING and DANGER hazards are logged into event history."""
        # Create site assessment with gas DANGER and noise WARNING
        readings = {
            "temperature": 24.0,
            "humidity": 45.0,
            "dust": 28.0,
            "noise": 78.0,  # WARNING
            "gas": 65.0,    # DANGER
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, simulator.parameters_meta)
        new_events = tracker.record_events_from_assessment(site_ass, tick=1)

        assert len(new_events) == 2
        assert tracker.event_count == 2
        assert tracker.has_events is True

        df = tracker.get_events_dataframe()
        assert len(df) == 2
        # Verify columns
        assert "Observed Value" in df.columns
        assert "Recommended Action" in df.columns

    def test_prevent_duplicate_events_on_static_refresh(self, tracker, simulator):
        """Static UI reruns with the same assessment do NOT create duplicate event log rows."""
        readings = {
            "temperature": 24.0,
            "humidity": 45.0,
            "dust": 28.0,
            "noise": 80.0,  # WARNING
            "gas": 8.0,
        }
        site_ass = HazardAssessmentEngine.assess_site(readings, simulator.parameters_meta)

        # First evaluation
        ev1 = tracker.record_events_from_assessment(site_ass, tick=1)
        assert len(ev1) == 1
        assert tracker.event_count == 1

        # Simulate 5 static UI reruns (same readings, same tick)
        for _ in range(5):
            ev_dup = tracker.record_events_from_assessment(site_ass, tick=1)
            assert len(ev_dup) == 0  # No duplicates logged!

        assert tracker.event_count == 1

    def test_event_logs_on_status_escalation(self, tracker, simulator):
        """Escalation from WARNING to DANGER creates a new event row."""
        # 1. Noise enters WARNING (78 dB)
        r1 = {"temperature": 24.0, "humidity": 45.0, "dust": 28.0, "noise": 78.0, "gas": 8.0}
        site_ass1 = HazardAssessmentEngine.assess_site(r1, simulator.parameters_meta)
        tracker.record_events_from_assessment(site_ass1, tick=1)
        assert tracker.event_count == 1

        # 2. Noise escalates to DANGER (92 dB)
        r2 = {"temperature": 24.0, "humidity": 45.0, "dust": 28.0, "noise": 92.0, "gas": 8.0}
        site_ass2 = HazardAssessmentEngine.assess_site(r2, simulator.parameters_meta)
        ev2 = tracker.record_events_from_assessment(site_ass2, tick=2)
        assert len(ev2) == 1
        assert tracker.event_count == 2
        assert "RED" in tracker.events[0].severity.color_label


# =============================================================================
# 3. Simulator Integration Tests
# =============================================================================
class TestSimulatorHistoryIntegration:
    def test_simulator_initializes_with_history(self, simulator):
        """EnvironmentalSimulator has integrated history tracker."""
        assert hasattr(simulator, "history")
        assert simulator.history is not None
        # On initialization or reset, baseline snapshot is recorded
        assert simulator.history.has_snapshots is True
        assert simulator.history.snapshot_count == 1

    def test_simulator_manual_update_records_history(self, simulator):
        """Manual set_reading records snapshot and logs hazard event."""
        simulator.set_reading("gas", 70.0)  # DANGER
        assert simulator.history.has_events is True
        events_df = simulator.get_hazard_events_df()
        assert len(events_df) >= 1
        assert "Gas Concentration" in events_df["Hazard Type"].values

    def test_simulator_step_records_history(self, simulator):
        """Automatic step_simulation records snapshots and tick count."""
        initial_count = simulator.history.snapshot_count
        simulator.step_simulation()
        assert simulator.history.snapshot_count == initial_count + 1

    def test_simulator_reset_clears_history(self, simulator):
        """Simulator reset restores clean baseline history."""
        simulator.set_reading("gas", 85.0)  # Triggers event
        assert simulator.history.has_events is True

        simulator.reset()
        # Reset clears events and restores initial baseline snapshot
        assert simulator.history.has_events is False
        assert simulator.history.snapshot_count == 1
        assert simulator.get_reading("gas").value == 8.0
