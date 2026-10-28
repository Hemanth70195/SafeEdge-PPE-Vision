"""
Video Processing Engine for SafeEdge PPE Vision.
Handles video ingestion, frame extraction, batch/stream inference, temporal tracking,
annotated video rendering, and statistical compliance report generation.
"""

import os
import time
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable, Union

import cv2
import numpy as np

from .ppe_detector import PPEDetector, Detection
from .ppe_assessor import PPEAssessor, WorkerAssessment, WorkerPPEStatus, PPEStatus
from .visualizer import PPEVisualizer
from .tracker import SimpleTracker, WorkerTrack
from .model_loader import ModelLoader

logger = logging.getLogger("SafeEdge.VideoProcessor")


@dataclass
class VideoMetadata:
    """
    Extracted metadata for an input video file.
    """
    file_path: str
    file_name: str
    file_size_mb: float
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    duration_str: str
    is_valid: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoProcessingStats:
    """
    Aggregated statistical summary and temporal records from video processing.
    """
    total_frames_in_video: int
    frames_processed: int
    sampling_stride: int
    total_person_observations: int
    unique_tracked_workers: int
    helmet_positive_observations: int
    vest_positive_observations: int
    helmet_violation_observations: int
    vest_violation_observations: int
    compliant_observations: int
    violation_observations: int
    review_required_observations: int
    mask_detected_observations: int
    processing_time_sec: float
    fps_throughput: float
    output_video_path: str
    timeline_data: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Format floats
        data["processing_time_sec"] = round(self.processing_time_sec, 2)
        data["fps_throughput"] = round(self.fps_throughput, 2)
        return data


def format_seconds(seconds: float) -> str:
    """Formats seconds into MM:SS.ms string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 100)
    return f"{mins:02d}:{secs:02d}.{millis:02d}"


def extract_video_metadata(video_path: Union[str, Path], max_size_mb: float = 500.0) -> VideoMetadata:
    """
    Validates and extracts technical metadata from a video file.

    Args:
        video_path: Path to the target video.
        max_size_mb: Configurable maximum allowable video file size in Megabytes.

    Returns:
        VideoMetadata dataclass instance.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        return VideoMetadata(
            file_path=str(path),
            file_name=path.name,
            file_size_mb=0.0,
            width=0,
            height=0,
            fps=0.0,
            total_frames=0,
            duration_sec=0.0,
            duration_str="00:00.00",
            is_valid=False,
            error_message=f"Video file does not exist at '{path}'.",
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return VideoMetadata(
            file_path=str(path),
            file_name=path.name,
            file_size_mb=round(file_size_mb, 2),
            width=0,
            height=0,
            fps=0.0,
            total_frames=0,
            duration_sec=0.0,
            duration_str="00:00.00",
            is_valid=False,
            error_message=f"Video size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb:.1f} MB.",
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoMetadata(
            file_path=str(path),
            file_name=path.name,
            file_size_mb=round(file_size_mb, 2),
            width=0,
            height=0,
            fps=0.0,
            total_frames=0,
            duration_sec=0.0,
            duration_str="00:00.00",
            is_valid=False,
            error_message="OpenCV could not open or decode video stream. File may be corrupted or in an unsupported format.",
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0.0 or np.isnan(fps):
            fps = 25.0  # Fallback to standard frame rate if unreadable

        if total_frames <= 0 or width <= 0 or height <= 0:
            return VideoMetadata(
                file_path=str(path),
                file_name=path.name,
                file_size_mb=round(file_size_mb, 2),
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_sec=0.0,
                duration_str="00:00.00",
                is_valid=False,
                error_message="Video contains zero valid frames or has invalid dimensions.",
            )

        duration_sec = total_frames / fps
        duration_str = format_seconds(duration_sec)

        return VideoMetadata(
            file_path=str(path),
            file_name=path.name,
            file_size_mb=round(file_size_mb, 2),
            width=width,
            height=height,
            fps=round(fps, 2),
            total_frames=total_frames,
            duration_sec=round(duration_sec, 2),
            duration_str=duration_str,
            is_valid=True,
        )

    finally:
        cap.release()


def convert_to_h264_mp4(src_path: str, dst_path: str) -> bool:
    """
    Attempts to re-encode video to H.264 (avc1) via bundled imageio-ffmpeg
    for guaranteed native HTML5 browser playback.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(src_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(dst_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return result.returncode == 0 and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    except Exception as e:
        logger.warning(f"H.264 re-encoding failed: {e}. Preserving OpenCV mp4v container.")
        return False


class VideoProcessor:
    """
    Core video processing pipeline for SafeEdge PPE Vision Phase 2.
    """

    def __init__(
        self,
        detector: Optional[PPEDetector] = None,
        assessor: Optional[PPEAssessor] = None,
        visualizer: Optional[PPEVisualizer] = None,
        tracker: Optional[SimpleTracker] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or ModelLoader.load_config()
        self.detector = detector or PPEDetector(config=self.config)
        self.assessor = assessor or PPEAssessor(config=self.config)
        self.visualizer = visualizer or PPEVisualizer(config=self.config)
        self.tracker = tracker or SimpleTracker()

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        frame_stride: int = 1,
        conf_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ) -> Tuple[str, VideoProcessingStats]:
        """
        Executes end-to-end PPE detection and compliance tracking across video frames.

        Args:
            video_path: Path to the input video.
            output_path: Destination path for annotated output video.
            frame_stride: Frame sampling stride (1 = process every frame, 2 = every 2nd frame, etc.).
            conf_threshold: Optional detection confidence threshold override.
            progress_callback: Callback func(processed_frames, total_frames, percent, status_msg).

        Returns:
            Tuple of (final_output_video_path, VideoProcessingStats).
        """
        metadata = extract_video_metadata(video_path)
        if not metadata.is_valid:
            raise ValueError(f"Invalid input video: {metadata.error_message}")

        input_path = Path(video_path).resolve()
        frame_stride = max(1, int(frame_stride))

        # Setup output destination
        if output_path is None:
            out_dir = Path(self.config.get("output", {}).get("default_output_dir", "output")).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            target_out_path = out_dir / f"annotated_{input_path.stem}.mp4"
        else:
            target_out_path = Path(output_path).resolve()
            target_out_path.parent.mkdir(parents=True, exist_ok=True)

        raw_cv_out_path = target_out_path.with_name(f"temp_{target_out_path.name}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video capture for: '{input_path}'")

        fps = metadata.fps
        width = metadata.width
        height = metadata.height
        total_frames = metadata.total_frames

        # Output video frame rate adjusted for sampling to maintain real-time duration
        output_fps = fps / frame_stride if frame_stride > 1 else fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_cv_out_path), fourcc, output_fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to initialize VideoWriter at '{raw_cv_out_path}'.")

        # Tracking and Aggregated Counters
        self.tracker = SimpleTracker()  # Fresh tracker per video
        frames_processed = 0
        total_person_observations = 0
        helmet_positive_obs = 0
        vest_positive_obs = 0
        helmet_violation_obs = 0
        vest_violation_obs = 0
        compliant_obs = 0
        violation_obs = 0
        review_req_obs = 0
        mask_detected_obs = 0

        timeline_data: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []

        start_time = time.time()
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_stride == 0:
                    current_time_sec = frame_idx / fps
                    time_str = format_seconds(current_time_sec)

                    # 1. Run PPE detection
                    detections = self.detector.detect(frame, conf_threshold=conf_threshold)

                    # Count mask observations (informational, not mandatory for compliance)
                    masks_in_frame = sum(1 for d in detections if d.category == "mask")
                    mask_detected_obs += masks_in_frame

                    # 2. Assess PPE compliance
                    raw_assessments = self.assessor.assess(detections, image_shape=(height, width))

                    # 3. Associate with persistent worker tracks
                    assessments = self.tracker.update(raw_assessments, frame_idx)

                    # 4. Annotate frame
                    annotated_frame = self.visualizer.visualize(frame, detections, assessments)
                    writer.write(annotated_frame)

                    # 5. Accumulate frame statistics
                    num_persons = len(assessments)
                    total_person_observations += num_persons

                    f_compliant = 0
                    f_violation = 0
                    f_review = 0

                    for a in assessments:
                        if a.overall_status == WorkerPPEStatus.COMPLIANT:
                            compliant_obs += 1
                            f_compliant += 1
                        elif a.overall_status == WorkerPPEStatus.VIOLATION:
                            violation_obs += 1
                            f_violation += 1
                        else:
                            review_req_obs += 1
                            f_review += 1

                        if a.helmet_status == PPEStatus.DETECTED:
                            helmet_positive_obs += 1
                        elif a.helmet_status == PPEStatus.MISSING:
                            helmet_violation_obs += 1

                        if a.vest_status == PPEStatus.DETECTED:
                            vest_positive_obs += 1
                        elif a.vest_status == PPEStatus.MISSING:
                            vest_violation_obs += 1

                        # Log individual worker event record
                        events.append({
                            "timestamp_str": time_str,
                            "timestamp_sec": round(current_time_sec, 2),
                            "frame_idx": frame_idx,
                            "worker_id": a.worker_id,
                            "overall_status": a.overall_status.value,
                            "helmet_status": a.helmet_status.value,
                            "vest_status": a.vest_status.value,
                            "person_confidence": round(a.person_confidence, 4),
                            "helmet_confidence": round(a.confidence_scores.get("helmet", 0.0), 4),
                            "vest_confidence": round(a.confidence_scores.get("vest", 0.0), 4),
                            "explanation": a.explanation,
                        })

                    # Record timeline data point for charts
                    timeline_data.append({
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(current_time_sec, 2),
                        "timestamp_str": time_str,
                        "persons_count": num_persons,
                        "compliant_count": f_compliant,
                        "violation_count": f_violation,
                        "review_count": f_review,
                        "mask_count": masks_in_frame,
                    })

                    frames_processed += 1

                    # Trigger progress callback
                    if progress_callback is not None:
                        percent = min(1.0, frame_idx / max(1, total_frames))
                        status_msg = f"Processing frame {frame_idx + 1}/{total_frames} ({frames_processed} sampled)"
                        progress_callback(frame_idx + 1, total_frames, percent, status_msg)

                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        elapsed_sec = max(0.001, time.time() - start_time)
        fps_throughput = frames_processed / elapsed_sec

        # Attempt H.264 re-encoding for universal browser playback
        final_video_path = target_out_path
        h264_success = convert_to_h264_mp4(str(raw_cv_out_path), str(target_out_path))
        if h264_success and raw_cv_out_path.exists():
            try:
                raw_cv_out_path.unlink()
            except Exception:
                pass
        else:
            # Fallback to OpenCV output
            if raw_cv_out_path.exists():
                if target_out_path.exists():
                    target_out_path.unlink()
                raw_cv_out_path.rename(target_out_path)

        stats = VideoProcessingStats(
            total_frames_in_video=total_frames,
            frames_processed=frames_processed,
            sampling_stride=frame_stride,
            total_person_observations=total_person_observations,
            unique_tracked_workers=self.tracker.total_unique_workers,
            helmet_positive_observations=helmet_positive_obs,
            vest_positive_observations=vest_positive_obs,
            helmet_violation_observations=helmet_violation_obs,
            vest_violation_observations=vest_violation_obs,
            compliant_observations=compliant_obs,
            violation_observations=violation_obs,
            review_required_observations=review_req_obs,
            mask_detected_observations=mask_detected_obs,
            processing_time_sec=elapsed_sec,
            fps_throughput=fps_throughput,
            output_video_path=str(final_video_path),
            timeline_data=timeline_data,
            events=events,
        )

        logger.info(
            f"Video processing complete: {frames_processed} frames analyzed in {elapsed_sec:.2f}s "
            f"({fps_throughput:.1f} fps). Tracked {self.tracker.total_unique_workers} unique workers."
        )

        return str(final_video_path), stats


def export_csv_report(stats: VideoProcessingStats, csv_path: Union[str, Path]) -> str:
    """
    Exports a structured CSV audit report of all observed worker events.

    Args:
        stats: VideoProcessingStats object from processing.
        csv_path: Destination path for the CSV report.

    Returns:
        Path to the saved CSV file.
    """
    import csv
    out_path = Path(csv_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_str",
        "timestamp_sec",
        "frame_idx",
        "worker_id",
        "overall_status",
        "helmet_status",
        "vest_status",
        "person_confidence",
        "helmet_confidence",
        "vest_confidence",
        "explanation",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for evt in stats.events:
            writer.writerow({
                "timestamp_str": evt.get("timestamp_str", ""),
                "timestamp_sec": evt.get("timestamp_sec", 0.0),
                "frame_idx": evt.get("frame_idx", 0),
                "worker_id": evt.get("worker_id", ""),
                "overall_status": evt.get("overall_status", ""),
                "helmet_status": evt.get("helmet_status", ""),
                "vest_status": evt.get("vest_status", ""),
                "person_confidence": evt.get("person_confidence", 0.0),
                "helmet_confidence": evt.get("helmet_confidence", 0.0),
                "vest_confidence": evt.get("vest_confidence", 0.0),
                "explanation": evt.get("explanation", ""),
            })

    return str(out_path)


def export_json_report(
    metadata: VideoMetadata,
    stats: VideoProcessingStats,
    json_path: Union[str, Path],
    model_info: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Exports a comprehensive JSON compliance report including metadata, statistics, and events.
    """
    out_path = Path(json_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "report_title": "SafeEdge PPE Vision - Construction Video Analysis Report",
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_metadata": metadata.to_dict(),
        "model_information": model_info or {
            "model_identifier": "Hansung-Cho/yolov8-ppe-detection",
            "weights_file": "models/best.pt",
            "base_architecture": "YOLOv8n",
            "license": "MIT",
        },
        "processing_settings": settings or {
            "sampling_stride": stats.sampling_stride,
            "output_format": "MP4 (H.264/mp4v)",
        },
        "aggregated_statistics": {
            "total_frames_in_video": stats.total_frames_in_video,
            "frames_processed": stats.frames_processed,
            "total_person_observations": stats.total_person_observations,
            "unique_tracked_workers": stats.unique_tracked_workers,
            "compliant_observations": stats.compliant_observations,
            "violation_observations": stats.violation_observations,
            "review_required_observations": stats.review_required_observations,
            "helmet_positive_observations": stats.helmet_positive_observations,
            "vest_positive_observations": stats.vest_positive_observations,
            "helmet_violation_observations": stats.helmet_violation_observations,
            "vest_violation_observations": stats.vest_violation_observations,
            "mask_detected_observations": stats.mask_detected_observations,
            "processing_time_sec": round(stats.processing_time_sec, 2),
            "fps_throughput": round(stats.fps_throughput, 2),
        },
        "timeline_data": stats.timeline_data,
        "event_records": stats.events,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    return str(out_path)
