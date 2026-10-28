"""
PPE Compliance Assessor for SafeEdge PPE Vision.
Associates PPE detections with workers using spatial reasoning and determines compliance.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .ppe_detector import Detection
from .model_loader import ModelLoader

logger = logging.getLogger("SafeEdge.PPEAssessor")


class PPEStatus(str, Enum):
    DETECTED = "DETECTED"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"


class WorkerPPEStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class WorkerAssessment:
    """
    Assessment result for an individual worker.
    """
    worker_id: str
    person_box: List[float]               # [x1, y1, x2, y2]
    person_confidence: float
    helmet_status: PPEStatus
    vest_status: PPEStatus
    overall_status: WorkerPPEStatus
    confidence_scores: Dict[str, float]   # {"person": 0.85, "helmet": 0.91, "vest": 0.78}
    explanation: str
    associated_detections: Dict[str, Optional[Detection]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "person_box": [round(c, 2) for c in self.person_box],
            "person_confidence": round(self.person_confidence, 4),
            "helmet_status": self.helmet_status.value,
            "vest_status": self.vest_status.value,
            "overall_status": self.overall_status.value,
            "confidence_scores": {k: round(v, 4) for k, v in self.confidence_scores.items()},
            "explanation": self.explanation,
        }


class PPEAssessor:
    """
    Evaluates worker PPE compliance by spatially associating helmet and vest detections.
    Adheres strictly to safety assessment rules:
    - Never assumes an undetected item is definitely missing if evidence is ambiguous.
    - Never associates PPE across overlapping workers without adequate evidence.
    - Prefers REVIEW_REQUIRED whenever occlusion, low resolution, or ambiguity occurs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ModelLoader.load_config()
        assessment_cfg = self.config.get("assessment", {})

        self.min_person_conf = float(assessment_cfg.get("min_person_confidence", 0.35))
        self.min_ppe_conf = float(assessment_cfg.get("min_ppe_confidence", 0.30))
        self.helmet_head_ratio = float(assessment_cfg.get("helmet_head_region_ratio", 0.40))
        self.vest_min_ratio = float(assessment_cfg.get("vest_torso_region_min_ratio", 0.15))
        self.vest_max_ratio = float(assessment_cfg.get("vest_torso_region_max_ratio", 0.85))
        self.min_h_overlap_ratio = float(assessment_cfg.get("min_horizontal_overlap_ratio", 0.25))
        self.worker_overlap_iou_thresh = float(assessment_cfg.get("worker_overlap_ambiguity_iou", 0.40))
        self.min_worker_height = float(assessment_cfg.get("min_worker_height_px", 50))
        self.min_worker_width = float(assessment_cfg.get("min_worker_width_px", 25))

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Computes Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union_area = area1 + area2 - intersection_area

        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    def get_head_zone(self, person_box: List[float]) -> List[float]:
        """Returns the spatial bounding box for the worker's head region."""
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        # Allow slight upward tolerance for helmet crowns above the detector's person box
        hy1 = py1 - 0.15 * ph
        hy2 = py1 + self.helmet_head_ratio * ph
        hx1 = px1 - 0.15 * pw
        hx2 = px2 + 0.15 * pw
        return [hx1, hy1, hx2, hy2]

    def get_torso_zone(self, person_box: List[float]) -> List[float]:
        """Returns the spatial bounding box for the worker's torso region."""
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        vy1 = py1 + self.vest_min_ratio * ph
        vy2 = py1 + self.vest_max_ratio * ph
        # Allow generous horizontal tolerance for vests that extend beyond person box
        vx1 = px1 - 0.30 * pw
        vx2 = px2 + 0.30 * pw
        return [vx1, vy1, vx2, vy2]

    def get_expanded_torso_zone(self, person_box: List[float]) -> List[float]:
        """Returns a larger fallback torso zone for vest IoU-overlap association."""
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        # Vests can start from shoulder level and extend to below waist
        vy1 = py1
        vy2 = py2
        vx1 = px1 - 0.40 * pw
        vx2 = px2 + 0.40 * pw
        return [vx1, vy1, vx2, vy2]

    def _calculate_spatial_fit(self, item: Detection, zone: List[float], person_box: List[float]) -> float:
        """
        Calculates a spatial fitness score [0.0, 1.0] representing how well a PPE detection
        fits within a worker's target zone.
        """
        cx, cy = item.center_x, item.center_y
        zx1, zy1, zx2, zy2 = zone

        # Allow center to be within a 20% margin of the zone (handles large vest boxes)
        zone_w = max(zx2 - zx1, 1.0)
        zone_h = max(zy2 - zy1, 1.0)
        margin_x = 0.20 * zone_w
        margin_y = 0.20 * zone_h
        if not (zx1 - margin_x <= cx <= zx2 + margin_x and zy1 - margin_y <= cy <= zy2 + margin_y):
            return 0.0

        # Horizontal overlap with person (relaxed for vests)
        px1, _, px2, _ = person_box
        ix_left = max(item.x1, px1)
        ix_right = min(item.x2, px2)
        h_overlap = max(0.0, ix_right - ix_left)
        person_w = max(px2 - px1, 1.0)
        if item.width > 0:
            h_ratio = h_overlap / item.width
            # Use person-relative overlap as alternative — vest may be wider than person box
            h_ratio_person = h_overlap / person_w
            h_ratio = max(h_ratio, h_ratio_person)
            if h_ratio < self.min_h_overlap_ratio:
                return 0.0
        else:
            h_ratio = 1.0

        # Calculate area of item inside zone
        ix1 = max(item.x1, zx1)
        iy1 = max(item.y1, zy1)
        ix2 = min(item.x2, zx2)
        iy2 = min(item.y2, zy2)

        if ix2 <= ix1 or iy2 <= iy1 or item.area <= 0:
            return 0.0

        in_zone_area = (ix2 - ix1) * (iy2 - iy1)
        containment = in_zone_area / item.area

        # Weighted score: containment * confidence * horizontal alignment
        return float(containment * item.confidence * (0.5 + 0.5 * h_ratio))

    def _calculate_vest_iou_fit(self, vest: Detection, person_box: List[float]) -> float:
        """
        Fallback: if standard spatial fit fails, use IoU between vest box and
        expanded person torso zone to catch large/overlapping vest detections.
        """
        expanded_zone = self.get_expanded_torso_zone(person_box)
        iou = self.compute_iou(vest.bbox, expanded_zone)
        if iou > 0.05:  # Vest meaningfully overlaps the person's body area
            return float(iou * vest.confidence)
        return 0.0

    def _find_overlapping_workers(self, persons: List[Detection]) -> Set[int]:
        """Identifies indices of workers who have high spatial overlap with another worker."""
        overlapping_indices: Set[int] = set()
        n = len(persons)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self.compute_iou(persons[i].bbox, persons[j].bbox)
                if iou >= self.worker_overlap_iou_thresh:
                    overlapping_indices.add(i)
                    overlapping_indices.add(j)
        return overlapping_indices

    def assess(
        self,
        detections: List[Detection],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[WorkerAssessment]:
        """
        Performs PPE assessment on all detected workers in an image.

        Args:
            detections: List of Detection objects from PPEDetector.
            image_shape: Optional (height, width) of the original image for edge/occlusion checks.

        Returns:
            List of WorkerAssessment objects for each detected worker.
        """
        persons = [d for d in detections if d.category == "person"]
        helmets = [d for d in detections if d.category == "helmet"]
        vests = [d for d in detections if d.category == "vest"]
        no_helmets = [d for d in detections if d.category == "no_helmet"]
        no_vests = [d for d in detections if d.category == "no_vest"]

        if not persons:
            return []

        overlapping_worker_indices = self._find_overlapping_workers(persons)

        # Pre-calculate candidate associations for helmets and vests across workers
        # item_id -> list of (worker_idx, fit_score)
        helmet_candidates: Dict[int, List[Tuple[int, float]]] = {}
        for h_idx, h in enumerate(helmets):
            if h.confidence < self.min_ppe_conf:
                continue
            candidates = []
            for p_idx, p in enumerate(persons):
                zone = self.get_head_zone(p.bbox)
                score = self._calculate_spatial_fit(h, zone, p.bbox)
                if score > 0.0:
                    candidates.append((p_idx, score))
            helmet_candidates[h_idx] = sorted(candidates, key=lambda x: x[1], reverse=True)

        vest_candidates: Dict[int, List[Tuple[int, float]]] = {}
        for v_idx, v in enumerate(vests):
            if v.confidence < self.min_ppe_conf:
                continue
            candidates = []
            for p_idx, p in enumerate(persons):
                zone = self.get_torso_zone(p.bbox)
                score = self._calculate_spatial_fit(v, zone, p.bbox)
                if score <= 0.0:
                    # Fallback: IoU-based association for large/wide vest detections
                    score = self._calculate_vest_iou_fit(v, p.bbox)
                if score > 0.0:
                    candidates.append((p_idx, score))
            vest_candidates[v_idx] = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Pre-calculate candidate associations for negative classes
        no_helmet_candidates: Dict[int, List[Tuple[int, float]]] = {}
        for nh_idx, nh in enumerate(no_helmets):
            candidates = []
            for p_idx, p in enumerate(persons):
                zone = self.get_head_zone(p.bbox)
                score = self._calculate_spatial_fit(nh, zone, p.bbox)
                if score > 0.0:
                    candidates.append((p_idx, score))
            no_helmet_candidates[nh_idx] = sorted(candidates, key=lambda x: x[1], reverse=True)

        no_vest_candidates: Dict[int, List[Tuple[int, float]]] = {}
        for nv_idx, nv in enumerate(no_vests):
            candidates = []
            for p_idx, p in enumerate(persons):
                zone = self.get_torso_zone(nv.bbox)
                score = self._calculate_spatial_fit(nv, zone, p.bbox)
                if score > 0.0:
                    candidates.append((p_idx, score))
            no_vest_candidates[nv_idx] = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Resolve associations while strictly respecting Rule 5:
        # "Never associate one worker's helmet or vest with another worker without adequate evidence."
        # If an item fits multiple workers with close scores, mark both as ambiguous.
        worker_helmets: Dict[int, Optional[Detection]] = {i: None for i in range(len(persons))}
        worker_helmet_ambiguous: Set[int] = set()

        for h_idx, cands in helmet_candidates.items():
            if not cands:
                continue
            h_det = helmets[h_idx]
            if len(cands) == 1:
                p_idx, _ = cands[0]
                if worker_helmets[p_idx] is None:
                    worker_helmets[p_idx] = h_det
                elif h_det.confidence > worker_helmets[p_idx].confidence:
                    worker_helmets[p_idx] = h_det
            else:
                top_p, top_score = cands[0]
                second_p, second_score = cands[1]
                # If top candidate is not clearly dominant (> 1.4x second), flag ambiguity
                if top_score < 1.4 * second_score or (top_p in overlapping_worker_indices and second_p in overlapping_worker_indices):
                    worker_helmet_ambiguous.add(top_p)
                    worker_helmet_ambiguous.add(second_p)
                else:
                    if worker_helmets[top_p] is None or h_det.confidence > worker_helmets[top_p].confidence:
                        worker_helmets[top_p] = h_det

        worker_vests: Dict[int, Optional[Detection]] = {i: None for i in range(len(persons))}
        worker_vest_ambiguous: Set[int] = set()

        for v_idx, cands in vest_candidates.items():
            if not cands:
                continue
            v_det = vests[v_idx]
            if len(cands) == 1:
                p_idx, _ = cands[0]
                if worker_vests[p_idx] is None:
                    worker_vests[p_idx] = v_det
                elif v_det.confidence > worker_vests[p_idx].confidence:
                    worker_vests[p_idx] = v_det
            else:
                top_p, top_score = cands[0]
                second_p, second_score = cands[1]
                if top_score < 1.4 * second_score or (top_p in overlapping_worker_indices and second_p in overlapping_worker_indices):
                    worker_vest_ambiguous.add(top_p)
                    worker_vest_ambiguous.add(second_p)
                else:
                    if worker_vests[top_p] is None or v_det.confidence > worker_vests[top_p].confidence:
                        worker_vests[top_p] = v_det

        # Associate negative classes (NO-Hardhat, NO-Safety Vest)
        worker_no_helmets: Dict[int, Optional[Detection]] = {i: None for i in range(len(persons))}
        for nh_idx, cands in no_helmet_candidates.items():
            if cands:
                p_idx, _ = cands[0]
                worker_no_helmets[p_idx] = no_helmets[nh_idx]

        worker_no_vests: Dict[int, Optional[Detection]] = {i: None for i in range(len(persons))}
        for nv_idx, cands in no_vest_candidates.items():
            if cands:
                p_idx, _ = cands[0]
                worker_no_vests[p_idx] = no_vests[nv_idx]

        # Build final WorkerAssessment for each person
        assessments: List[WorkerAssessment] = []
        img_h, img_w = image_shape if image_shape else (100000, 100000)

        for p_idx, person in enumerate(persons):
            worker_id = f"worker_{p_idx + 1}"
            pw = person.width
            ph = person.height
            px1, py1, px2, py2 = person.bbox
            conf_scores = {"person": person.confidence}

            is_overlapping = p_idx in overlapping_worker_indices
            is_low_res = ph < self.min_worker_height or pw < self.min_worker_width
            is_low_conf = person.confidence < self.min_person_conf

            # Boundary clipping check: person head or torso truncated at image boundary
            head_truncated = py1 <= 5.0
            torso_truncated = py2 >= (img_h - 5.0)

            helmet_det = worker_helmets[p_idx]
            vest_det = worker_vests[p_idx]
            no_helmet_det = worker_no_helmets[p_idx]
            no_vest_det = worker_no_vests[p_idx]

            associated: Dict[str, Optional[Detection]] = {
                "person": person,
                "helmet": helmet_det,
                "vest": vest_det,
                "no_helmet": no_helmet_det,
                "no_vest": no_vest_det,
            }

            # -------------------------------------------------------------
            # Evaluate Helmet Status
            # -------------------------------------------------------------
            if helmet_det is not None and p_idx not in worker_helmet_ambiguous:
                helmet_status = PPEStatus.DETECTED
                conf_scores["helmet"] = helmet_det.confidence
            elif no_helmet_det is not None:
                # Strong evidence of violation: explicitly detected NO-Hardhat
                helmet_status = PPEStatus.MISSING
                conf_scores["no_helmet"] = no_helmet_det.confidence
            elif p_idx in worker_helmet_ambiguous or is_overlapping:
                # Ambiguous spatial association
                helmet_status = PPEStatus.UNKNOWN
            elif is_low_res or head_truncated or is_low_conf:
                # Rule 4: Never assume an undetected helmet is definitely missing when resolution/visibility is compromised
                helmet_status = PPEStatus.UNKNOWN
            else:
                # Worker is clear, confident, head is visible, no ambiguity: absence supported
                helmet_status = PPEStatus.MISSING

            # -------------------------------------------------------------
            # Evaluate Vest Status
            # -------------------------------------------------------------
            if vest_det is not None and p_idx not in worker_vest_ambiguous:
                vest_status = PPEStatus.DETECTED
                conf_scores["vest"] = vest_det.confidence
            elif no_vest_det is not None:
                # Strong evidence of violation: explicitly detected NO-Safety Vest
                vest_status = PPEStatus.MISSING
                conf_scores["no_vest"] = no_vest_det.confidence
            elif p_idx in worker_vest_ambiguous or is_overlapping:
                vest_status = PPEStatus.UNKNOWN
            elif is_low_res or torso_truncated or is_low_conf:
                # Rule 4: Never assume an undetected vest is definitely missing when resolution/visibility is compromised
                vest_status = PPEStatus.UNKNOWN
            else:
                # Worker is clear, confident, torso is visible, no ambiguity: absence supported
                vest_status = PPEStatus.MISSING

            # -------------------------------------------------------------
            # Determine Overall PPE Status and Explanation
            # Rules:
            # 1. COMPLIANT: Both required PPE items are DETECTED with sufficient confidence.
            # 2. VIOLATION: At least one required PPE item is sufficiently supported as MISSING.
            # 3. REVIEW_REQUIRED: Evidence is ambiguous, worker is occluded, detections are unreliable or model cannot make a reliable determination.
            # -------------------------------------------------------------
            reasons: List[str] = []

            if is_low_conf:
                overall_status = WorkerPPEStatus.REVIEW_REQUIRED
                reasons.append(f"Person detection confidence ({person.confidence:.2f}) below threshold ({self.min_person_conf:.2f})")
            elif is_overlapping:
                overall_status = WorkerPPEStatus.REVIEW_REQUIRED
                reasons.append("Worker overlaps significantly with another worker; spatial association is ambiguous")
            elif is_low_res:
                overall_status = WorkerPPEStatus.REVIEW_REQUIRED
                reasons.append(f"Worker bounding box size ({int(pw)}x{int(ph)}px) is too small for reliable PPE compliance verification")
            elif helmet_status == PPEStatus.MISSING or vest_status == PPEStatus.MISSING:
                overall_status = WorkerPPEStatus.VIOLATION
                missing_items = []
                if helmet_status == PPEStatus.MISSING:
                    missing_items.append("safety helmet")
                if vest_status == PPEStatus.MISSING:
                    missing_items.append("safety vest")
                reasons.append(f"Safety violation: Missing required {', '.join(missing_items)}")
            elif helmet_status == PPEStatus.DETECTED and vest_status == PPEStatus.DETECTED:
                overall_status = WorkerPPEStatus.COMPLIANT
                reasons.append("Fully compliant: Both safety helmet and high-visibility vest detected and verified")
            else:
                # At least one UNKNOWN and none MISSING
                overall_status = WorkerPPEStatus.REVIEW_REQUIRED
                unknown_items = []
                if helmet_status == PPEStatus.UNKNOWN:
                    unknown_items.append("helmet")
                if vest_status == PPEStatus.UNKNOWN:
                    unknown_items.append("vest")
                reasons.append(f"Manual review required: Inconclusive {', '.join(unknown_items)} visibility/association")

            # Append context notes to explanation
            if head_truncated:
                reasons.append("Head region truncated by image top edge")
            if torso_truncated:
                reasons.append("Torso region truncated by image bottom edge")
            if p_idx in worker_helmet_ambiguous or p_idx in worker_vest_ambiguous:
                reasons.append("PPE item candidates overlap with adjacent workers")

            explanation = "; ".join(reasons)

            assessment = WorkerAssessment(
                worker_id=worker_id,
                person_box=person.bbox,
                person_confidence=person.confidence,
                helmet_status=helmet_status,
                vest_status=vest_status,
                overall_status=overall_status,
                confidence_scores=conf_scores,
                explanation=explanation,
                associated_detections=associated,
            )
            assessments.append(assessment)

        return assessments
