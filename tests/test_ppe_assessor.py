"""
Automated Unit Tests for PPE Compliance Assessor (Mocked/Synthetic Scenarios).
Tests spatial association, compliance rules, edge cases, and ambiguity resolution.
"""

import pytest
from core.ppe_detector import Detection
from core.ppe_assessor import PPEAssessor, PPEStatus, WorkerPPEStatus


@pytest.fixture
def assessor():
    """Provides a PPEAssessor instance with standard test configuration."""
    return PPEAssessor()


def create_person(x1=100.0, y1=100.0, x2=200.0, y2=400.0, conf=0.90) -> Detection:
    """Helper to create a synthetic person detection."""
    return Detection(
        class_name="Person",
        confidence=conf,
        bbox=[x1, y1, x2, y2],
        category="person",
        class_id=5,
    )


def create_helmet(x1=120.0, y1=90.0, x2=180.0, y2=160.0, conf=0.88) -> Detection:
    """Helper to create a synthetic helmet detection positioned on the head."""
    return Detection(
        class_name="Hardhat",
        confidence=conf,
        bbox=[x1, y1, x2, y2],
        category="helmet",
        class_id=0,
    )


def create_vest(x1=110.0, y1=170.0, x2=190.0, y2=320.0, conf=0.85) -> Detection:
    """Helper to create a synthetic vest detection positioned on the torso."""
    return Detection(
        class_name="Safety Vest",
        confidence=conf,
        bbox=[x1, y1, x2, y2],
        category="vest",
        class_id=7,
    )


def test_fully_compliant_worker(assessor):
    """Test 1: Worker wearing both safety helmet and vest -> COMPLIANT."""
    person = create_person(100, 100, 200, 400)
    helmet = create_helmet(120, 95, 180, 160)
    vest = create_vest(110, 170, 190, 310)

    results = assessor.assess([person, helmet, vest], image_shape=(800, 800))
    assert len(results) == 1
    worker = results[0]
    assert worker.overall_status == WorkerPPEStatus.COMPLIANT
    assert worker.helmet_status == PPEStatus.DETECTED
    assert worker.vest_status == PPEStatus.DETECTED
    assert "Fully compliant" in worker.explanation


def test_worker_missing_helmet(assessor):
    """Test 2: Worker wearing vest but missing helmet -> VIOLATION."""
    person = create_person(100, 100, 200, 400)
    vest = create_vest(110, 170, 190, 310)

    results = assessor.assess([person, vest], image_shape=(800, 800))
    assert len(results) == 1
    worker = results[0]
    assert worker.overall_status == WorkerPPEStatus.VIOLATION
    assert worker.helmet_status == PPEStatus.MISSING
    assert worker.vest_status == PPEStatus.DETECTED
    assert "safety helmet" in worker.explanation.lower()


def test_worker_missing_vest(assessor):
    """Test 3: Worker wearing helmet but missing vest -> VIOLATION."""
    person = create_person(100, 100, 200, 400)
    helmet = create_helmet(120, 95, 180, 160)

    results = assessor.assess([person, helmet], image_shape=(800, 800))
    assert len(results) == 1
    worker = results[0]
    assert worker.overall_status == WorkerPPEStatus.VIOLATION
    assert worker.helmet_status == PPEStatus.DETECTED
    assert worker.vest_status == PPEStatus.MISSING
    assert "safety vest" in worker.explanation.lower()


def test_worker_with_explicit_negative_classes(assessor):
    """Test 4: Explicit NO-Hardhat and NO-Safety Vest detections -> VIOLATION."""
    person = create_person(100, 100, 200, 400)
    no_helmet = Detection("NO-Hardhat", 0.85, [120, 95, 180, 160], "no_helmet", 2)
    no_vest = Detection("NO-Safety Vest", 0.80, [110, 170, 190, 310], "no_vest", 4)

    results = assessor.assess([person, no_helmet, no_vest], image_shape=(800, 800))
    assert len(results) == 1
    worker = results[0]
    assert worker.overall_status == WorkerPPEStatus.VIOLATION
    assert worker.helmet_status == PPEStatus.MISSING
    assert worker.vest_status == PPEStatus.MISSING


def test_multiple_workers_independent_assessment(assessor):
    """Test 5: Multiple distinct workers evaluated independently."""
    # Worker 1: Compliant
    p1 = create_person(50, 100, 150, 400)
    h1 = create_helmet(70, 95, 130, 160)
    v1 = create_vest(60, 170, 140, 310)

    # Worker 2: Non-compliant (missing vest)
    p2 = create_person(400, 100, 500, 400)
    h2 = create_helmet(420, 95, 480, 160)

    results = assessor.assess([p1, p2, h1, v1, h2], image_shape=(800, 800))
    assert len(results) == 2

    w1 = results[0]
    w2 = results[1]
    assert w1.overall_status == WorkerPPEStatus.COMPLIANT
    assert w2.overall_status == WorkerPPEStatus.VIOLATION


def test_overlapping_and_ambiguous_workers(assessor):
    """Test 6: Overlapping workers produce REVIEW_REQUIRED rather than false violations."""
    # Two workers overlapping significantly (IoU > 0.40)
    p1 = create_person(100, 100, 200, 400)
    p2 = create_person(120, 100, 220, 400)  # Heavy overlap

    results = assessor.assess([p1, p2], image_shape=(800, 800))
    assert len(results) == 2
    assert results[0].overall_status == WorkerPPEStatus.REVIEW_REQUIRED
    assert results[1].overall_status == WorkerPPEStatus.REVIEW_REQUIRED
    assert "overlaps significantly" in results[0].explanation


def test_low_confidence_person_detection(assessor):
    """Test 7: Worker with low person detection confidence -> REVIEW_REQUIRED."""
    person = create_person(100, 100, 200, 400, conf=0.25)
    helmet = create_helmet(120, 95, 180, 160, conf=0.90)
    vest = create_vest(110, 170, 190, 310, conf=0.90)

    results = assessor.assess([person, helmet, vest], image_shape=(800, 800))
    assert len(results) == 1
    assert results[0].overall_status == WorkerPPEStatus.REVIEW_REQUIRED
    assert "confidence" in results[0].explanation.lower()


def test_occluded_worker_at_image_boundary(assessor):
    """Test 8: Worker truncated at top image boundary (head cut off) -> REVIEW_REQUIRED."""
    # y1 is 2.0 (truncated by top border 0)
    person = create_person(100, 2.0, 200, 300)
    vest = create_vest(110, 50, 190, 200)

    results = assessor.assess([person, vest], image_shape=(600, 600))
    assert len(results) == 1
    worker = results[0]
    # We cannot confirm helmet absence because the head is outside the camera view
    assert worker.helmet_status == PPEStatus.UNKNOWN
    assert worker.overall_status == WorkerPPEStatus.REVIEW_REQUIRED
    assert "top edge" in worker.explanation.lower()


def test_small_distant_worker_low_resolution(assessor):
    """Test 9: Very small/distant worker bounding box -> REVIEW_REQUIRED (Rule 4)."""
    # 20x30 px box: below minimum worker height (50px)
    tiny_person = create_person(100, 100, 120, 130)

    results = assessor.assess([tiny_person], image_shape=(800, 800))
    assert len(results) == 1
    worker = results[0]
    assert worker.helmet_status == PPEStatus.UNKNOWN
    assert worker.vest_status == PPEStatus.UNKNOWN
    assert worker.overall_status == WorkerPPEStatus.REVIEW_REQUIRED
    assert "too small" in worker.explanation.lower()


def test_empty_detections(assessor):
    """Test 10: Empty detections list -> returns empty assessment list."""
    results = assessor.assess([])
    assert results == []
