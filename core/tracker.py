"""
Worker Temporal Tracker for SafeEdge PPE Vision.
Tracks detected workers across video frames using spatial IoU matching.
Enables reliable counting of unique workers and consistent Worker IDs across the video.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from .ppe_assessor import WorkerAssessment, WorkerPPEStatus, PPEStatus


@dataclass
class WorkerTrack:
    """
    Persistent track of an individual worker across video frames.
    """
    track_id: int
    label: str                       # e.g. "Worker #1"
    bbox: List[float]                # Last known [x1, y1, x2, y2]
    first_seen_frame: int
    last_seen_frame: int
    frames_seen: int = 0
    missed_frames: int = 0
    total_compliant: int = 0
    total_violations: int = 0
    total_reviews: int = 0
    helmet_positive_count: int = 0
    vest_positive_count: int = 0
    last_status: Optional[WorkerPPEStatus] = None

    def update(self, assessment: WorkerAssessment, frame_idx: int):
        """Updates track with a new frame observation."""
        self.bbox = assessment.person_box
        self.last_seen_frame = frame_idx
        self.frames_seen += 1
        self.missed_frames = 0
        self.last_status = assessment.overall_status

        if assessment.overall_status == WorkerPPEStatus.COMPLIANT:
            self.total_compliant += 1
        elif assessment.overall_status == WorkerPPEStatus.VIOLATION:
            self.total_violations += 1
        else:
            self.total_reviews += 1

        if assessment.helmet_status == PPEStatus.DETECTED:
            self.helmet_positive_count += 1
        if assessment.vest_status == PPEStatus.DETECTED:
            self.vest_positive_count += 1


class SimpleTracker:
    """
    Lightweight, deterministic IoU-based tracker for worker bounding boxes across frames.
    Distinguishes unique workers from repeated frame-level observations.
    """

    def __init__(self, iou_threshold: float = 0.30, max_lost_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.active_tracks: List[WorkerTrack] = []
        self.all_tracks: List[WorkerTrack] = []

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Computes IoU between two boxes [x1, y1, x2, y2]."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, assessments: List[WorkerAssessment], frame_idx: int) -> List[WorkerAssessment]:
        """
        Associates current frame worker assessments with existing tracks.
        Assigns persistent worker IDs to each assessment and updates track history.

        Args:
            assessments: Worker assessments for the current frame.
            frame_idx: Index of the current video frame.

        Returns:
            Updated assessments list with consistent worker_ids.
        """
        if not assessments:
            # Mark all active tracks as missed for this frame
            for track in self.active_tracks:
                track.missed_frames += 1
            # Prune lost tracks
            self.active_tracks = [t for t in self.active_tracks if t.missed_frames <= self.max_lost_frames]
            return []

        # If no active tracks, initialize all detections as new tracks
        if not self.active_tracks:
            for a in assessments:
                track = WorkerTrack(
                    track_id=self.next_track_id,
                    label=f"Worker #{self.next_track_id}",
                    bbox=a.person_box,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                )
                track.update(a, frame_idx)
                a.worker_id = track.label
                self.next_track_id += 1
                self.active_tracks.append(track)
                self.all_tracks.append(track)
            return assessments

        # Compute IoU cost matrix between active tracks and current assessments
        num_tracks = len(self.active_tracks)
        num_dets = len(assessments)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for t_idx, track in enumerate(self.active_tracks):
            for d_idx, a in enumerate(assessments):
                iou_matrix[t_idx, d_idx] = self.compute_iou(track.bbox, a.person_box)

        # Greedy bipartite matching
        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        # Find best matches in descending order of IoU
        while True:
            if len(matched_tracks) == num_tracks or len(matched_dets) == num_dets:
                break
            best_iou = -1.0
            best_t, best_d = -1, -1

            for t_idx in range(num_tracks):
                if t_idx in matched_tracks:
                    continue
                for d_idx in range(num_dets):
                    if d_idx in matched_dets:
                        continue
                    if iou_matrix[t_idx, d_idx] > best_iou:
                        best_iou = iou_matrix[t_idx, d_idx]
                        best_t, best_d = t_idx, d_idx

            if best_iou < self.iou_threshold:
                break

            matched_tracks.add(best_t)
            matched_dets.add(best_d)

            track = self.active_tracks[best_t]
            a = assessments[best_d]
            track.update(a, frame_idx)
            a.worker_id = track.label

        # Handle unmatched detections -> New tracks
        for d_idx, a in enumerate(assessments):
            if d_idx not in matched_dets:
                track = WorkerTrack(
                    track_id=self.next_track_id,
                    label=f"Worker #{self.next_track_id}",
                    bbox=a.person_box,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                )
                track.update(a, frame_idx)
                a.worker_id = track.label
                self.next_track_id += 1
                self.active_tracks.append(track)
                self.all_tracks.append(track)

        # Handle unmatched tracks -> Increment missed
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_tracks:
                track.missed_frames += 1

        # Prune expired tracks
        self.active_tracks = [t for t in self.active_tracks if t.missed_frames <= self.max_lost_frames]

        return assessments

    @property
    def total_unique_workers(self) -> int:
        """Returns the total number of distinct workers tracked."""
        return len(self.all_tracks)

    def get_tracks(self) -> List[WorkerTrack]:
        """Returns all tracked workers."""
        return list(self.all_tracks)
