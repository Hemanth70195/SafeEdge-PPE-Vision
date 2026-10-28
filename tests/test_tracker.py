"""
Automated Unit Tests for Worker Temporal Tracker.
Validates tracking consistency, unique worker counting, and observation accumulation.
"""

import pytest
from core.ppe_assessor import WorkerAssessment, PPEStatus, WorkerPPEStatus
from core.tracker import SimpleTracker, WorkerTrack


def make_assessment(x1=100.0, y1=100.0, x2=200.0, y2=400.0, status=WorkerPPEStatus.COMPLIANT):
    return WorkerAssessment(
        worker_id="temp",
        person_box=[x1, y1, x2, y2],
        person_confidence=0.90,
        helmet_status=PPEStatus.DETECTED if status == WorkerPPEStatus.COMPLIANT else PPEStatus.MISSING,
        vest_status=PPEStatus.DETECTED,
        overall_status=status,
        confidence_scores={"person": 0.90, "helmet": 0.85, "vest": 0.80},
        explanation="Test assessment",
    )


def test_tracker_assigns_consistent_ids_across_frames():
    """Verify that a worker moving slightly between frames retains the same Worker ID."""
    tracker = SimpleTracker(iou_threshold=0.30)

    # Frame 0
    a0 = make_assessment(100.0, 100.0, 200.0, 400.0)
    updated0 = tracker.update([a0], frame_idx=0)
    assert len(updated0) == 1
    assert updated0[0].worker_id == "Worker #1"

    # Frame 1 (slight movement: high IoU)
    a1 = make_assessment(102.0, 101.0, 202.0, 401.0)
    updated1 = tracker.update([a1], frame_idx=1)
    assert len(updated1) == 1
    assert updated1[0].worker_id == "Worker #1"

    # Frame 2
    a2 = make_assessment(105.0, 102.0, 205.0, 402.0)
    updated2 = tracker.update([a2], frame_idx=2)
    assert len(updated2) == 1
    assert updated2[0].worker_id == "Worker #1"

    # Total unique workers should be exactly 1, NOT 3!
    assert tracker.total_unique_workers == 1


def test_tracker_distinguishes_multiple_distinct_workers():
    """Verify distinct workers receive distinct persistent IDs."""
    tracker = SimpleTracker()

    # Frame 0: Two workers in different parts of the frame
    w1 = make_assessment(50.0, 100.0, 150.0, 400.0)
    w2 = make_assessment(500.0, 100.0, 600.0, 400.0)
    res0 = tracker.update([w1, w2], frame_idx=0)

    assert len(res0) == 2
    ids = {r.worker_id for r in res0}
    assert ids == {"Worker #1", "Worker #2"}
    assert tracker.total_unique_workers == 2


def test_tracker_accumulates_observation_statistics():
    """Verify track history correctly tallies frame observations and compliance states."""
    tracker = SimpleTracker()

    # Frame 0: Compliant
    tracker.update([make_assessment(status=WorkerPPEStatus.COMPLIANT)], frame_idx=0)
    # Frame 1: Violation
    tracker.update([make_assessment(status=WorkerPPEStatus.VIOLATION)], frame_idx=1)
    # Frame 2: Review Required
    tracker.update([make_assessment(status=WorkerPPEStatus.REVIEW_REQUIRED)], frame_idx=2)

    tracks = tracker.get_tracks()
    assert len(tracks) == 1
    t = tracks[0]
    assert t.frames_seen == 3
    assert t.total_compliant == 1
    assert t.total_violations == 1
    assert t.total_reviews == 1


def test_tracker_handles_empty_frames():
    """Verify tracker gracefully handles frames where no workers are detected."""
    tracker = SimpleTracker()

    tracker.update([make_assessment()], frame_idx=0)
    assert tracker.total_unique_workers == 1

    # Frame with no detections
    empty_res = tracker.update([], frame_idx=1)
    assert empty_res == []
    # Track persists but misses a frame
    assert tracker.total_unique_workers == 1
