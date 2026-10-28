"""
Automated Unit Tests for Video Processor.
Tests metadata extraction, error handling, frame sampling, mocked inference,
resource cleanup, and CSV/JSON report exports.
"""

import os
import json
import csv
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock

from core.ppe_detector import Detection, PPEDetector
from core.ppe_assessor import PPEAssessor, WorkerPPEStatus
from core.visualizer import PPEVisualizer
from core.tracker import SimpleTracker
from core.video_processor import (
    VideoProcessor,
    VideoMetadata,
    VideoProcessingStats,
    extract_video_metadata,
    export_csv_report,
    export_json_report,
    format_seconds,
)


def create_synthetic_video(file_path: Path, num_frames: int = 15, width: int = 320, height: int = 240, fps: float = 10.0):
    """Generates a small test video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_format_seconds_utility():
    """Verify string formatting of time values."""
    assert format_seconds(0.0) == "00:00.00"
    assert format_seconds(65.5) == "01:05.50"
    assert format_seconds(125.25) == "02:05.25"


def test_extract_metadata_valid_video(tmp_path):
    """Verify metadata extraction on a valid synthetic video file."""
    v_path = tmp_path / "valid_test.mp4"
    create_synthetic_video(v_path, num_frames=20, width=320, height=240, fps=10.0)

    meta = extract_video_metadata(v_path)
    assert meta.is_valid is True
    assert meta.width == 320
    assert meta.height == 240
    assert meta.total_frames == 20
    assert meta.fps == 10.0
    assert meta.duration_sec == 2.0
    assert meta.file_size_mb > 0.0


def test_extract_metadata_nonexistent_file():
    """Verify metadata extraction handles missing files gracefully."""
    meta = extract_video_metadata("non_existent_video_path.mp4")
    assert meta.is_valid is False
    assert "does not exist" in meta.error_message


def test_extract_metadata_corrupted_file(tmp_path):
    """Verify metadata extraction flags corrupted or invalid video files."""
    corrupt_path = tmp_path / "corrupt.mp4"
    with open(corrupt_path, "w") as f:
        f.write("This is not a real video stream.")

    meta = extract_video_metadata(corrupt_path)
    assert meta.is_valid is False
    assert meta.error_message is not None


def test_extract_metadata_exceeds_max_size(tmp_path):
    """Verify metadata extraction enforces maximum file size thresholds."""
    v_path = tmp_path / "size_test.mp4"
    create_synthetic_video(v_path, num_frames=5)

    # Set threshold lower than file size
    meta = extract_video_metadata(v_path, max_size_mb=0.000001)
    assert meta.is_valid is False
    assert "exceeds maximum allowed limit" in meta.error_message


def test_video_processor_mocked_end_to_end(tmp_path):
    """
    Test end-to-end video processing with mocked detector.
    Verifies frame processing, statistics aggregation, and output generation.
    """
    v_path = tmp_path / "stream_test.mp4"
    create_synthetic_video(v_path, num_frames=10, width=320, height=240, fps=10.0)

    # Mock detector returning 1 compliant worker on all frames
    mock_detector = MagicMock(spec=PPEDetector)
    mock_detections = [
        Detection("Person", 0.90, [50.0, 50.0, 150.0, 200.0], "person", 5),
        Detection("Hardhat", 0.88, [70.0, 45.0, 130.0, 100.0], "helmet", 0),
        Detection("Safety Vest", 0.85, [60.0, 90.0, 140.0, 170.0], "vest", 7),
    ]
    mock_detector.detect.return_value = mock_detections

    out_video_path = tmp_path / "output_annotated.mp4"

    processor = VideoProcessor(
        detector=mock_detector,
        assessor=PPEAssessor(),
        visualizer=PPEVisualizer(),
        tracker=SimpleTracker(),
    )

    out_path, stats = processor.process_video(
        video_path=v_path,
        output_path=out_video_path,
        frame_stride=1,
    )

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
    assert stats.total_frames_in_video == 10
    assert stats.frames_processed == 10
    assert stats.sampling_stride == 1
    assert stats.total_person_observations == 10
    # Exactly 1 unique worker tracked across all 10 frames
    assert stats.unique_tracked_workers == 1
    assert stats.compliant_observations == 10
    assert stats.violation_observations == 0
    assert len(stats.timeline_data) == 10
    assert len(stats.events) == 10


def test_video_processor_frame_sampling(tmp_path):
    """Verify that frame stride properly samples video frames."""
    v_path = tmp_path / "stride_test.mp4"
    create_synthetic_video(v_path, num_frames=12, fps=10.0)

    mock_detector = MagicMock(spec=PPEDetector)
    mock_detector.detect.return_value = []

    out_video_path = tmp_path / "sampled_output.mp4"
    processor = VideoProcessor(detector=mock_detector)

    # Stride of 3 on 12 frames should process exactly 4 frames (0, 3, 6, 9)
    out_path, stats = processor.process_video(
        video_path=v_path,
        output_path=out_video_path,
        frame_stride=3,
    )

    assert stats.frames_processed == 4
    assert stats.sampling_stride == 3
    assert len(stats.timeline_data) == 4


def test_export_csv_and_json_reports(tmp_path):
    """Verify CSV and JSON report generators produce valid, structured files."""
    stats = VideoProcessingStats(
        total_frames_in_video=30,
        frames_processed=15,
        sampling_stride=2,
        total_person_observations=15,
        unique_tracked_workers=1,
        helmet_positive_observations=15,
        vest_positive_observations=10,
        helmet_violation_observations=0,
        vest_violation_observations=5,
        compliant_observations=10,
        violation_observations=5,
        review_required_observations=0,
        mask_detected_observations=2,
        processing_time_sec=1.25,
        fps_throughput=12.0,
        output_video_path="output/test.mp4",
        timeline_data=[{"frame_idx": 0, "timestamp_sec": 0.0, "timestamp_str": "00:00.00", "persons_count": 1, "compliant_count": 1, "violation_count": 0, "review_count": 0, "mask_count": 0}],
        events=[{
            "timestamp_str": "00:00.00",
            "timestamp_sec": 0.0,
            "frame_idx": 0,
            "worker_id": "Worker #1",
            "overall_status": "COMPLIANT",
            "helmet_status": "DETECTED",
            "vest_status": "DETECTED",
            "person_confidence": 0.90,
            "helmet_confidence": 0.88,
            "vest_confidence": 0.85,
            "explanation": "Fully compliant",
        }],
    )

    # Test CSV Export
    csv_file = tmp_path / "test_report.csv"
    export_csv_report(stats, csv_file)
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["worker_id"] == "Worker #1"
        assert reader[0]["overall_status"] == "COMPLIANT"

    # Test JSON Export
    meta = VideoMetadata(
        file_path="test.mp4",
        file_name="test.mp4",
        file_size_mb=1.5,
        width=640,
        height=480,
        fps=30.0,
        total_frames=30,
        duration_sec=1.0,
        duration_str="00:01.00",
        is_valid=True,
    )
    json_file = tmp_path / "test_summary.json"
    export_json_report(meta, stats, json_file)
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["aggregated_statistics"]["unique_tracked_workers"] == 1
        assert len(data["event_records"]) == 1
