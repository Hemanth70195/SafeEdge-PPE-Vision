"""
Visualizer for SafeEdge PPE Vision.
Renders bounding boxes, worker compliance banners, labels, and color legends on OpenCV images.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from .ppe_detector import Detection
from .ppe_assessor import WorkerAssessment, WorkerPPEStatus, PPEStatus
from .model_loader import ModelLoader


class PPEVisualizer:
    """
    Renders structured PPE detections and compliance assessments onto OpenCV images.
    """

    # BGR Color definitions
    COLOR_COMPLIANT = (50, 205, 50)       # Green
    COLOR_VIOLATION = (30, 30, 220)       # Red
    COLOR_REVIEW = (0, 140, 255)          # Orange

    COLOR_HELMET = (0, 215, 255)          # Gold / Yellow-Orange
    COLOR_VEST = (0, 255, 127)            # Spring Green
    COLOR_NO_PPE = (0, 0, 255)            # Deep Red
    COLOR_DEFAULT = (200, 200, 200)       # Light Gray
    COLOR_TEXT_WHITE = (255, 255, 255)
    COLOR_TEXT_BLACK = (0, 0, 0)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ModelLoader.load_config()
        vis_cfg = self.config.get("visualization", {})
        self.font_scale = float(vis_cfg.get("font_scale", 0.55))
        self.thickness = int(vis_cfg.get("line_thickness", 2))
        self.show_legend = bool(vis_cfg.get("show_legend", True))

    def _get_status_color(self, status: WorkerPPEStatus) -> Tuple[int, int, int]:
        if status == WorkerPPEStatus.COMPLIANT:
            return self.COLOR_COMPLIANT
        elif status == WorkerPPEStatus.VIOLATION:
            return self.COLOR_VIOLATION
        else:
            return self.COLOR_REVIEW

    def _get_item_color(self, category: str) -> Tuple[int, int, int]:
        cat = category.lower()
        if cat == "helmet":
            return self.COLOR_HELMET
        elif cat == "vest":
            return self.COLOR_VEST
        elif cat in ["no_helmet", "no_vest", "no-hardhat", "no-safety vest"]:
            return self.COLOR_NO_PPE
        return self.COLOR_DEFAULT

    def draw_legend(self, img: np.ndarray, x: int = 15, y: int = 15) -> np.ndarray:
        """Draws a compact, readable compliance status legend in the corner."""
        legend_items = [
            ("COMPLIANT", self.COLOR_COMPLIANT),
            ("VIOLATION", self.COLOR_VIOLATION),
            ("REVIEW REQ", self.COLOR_REVIEW),
            ("Helmet Box", self.COLOR_HELMET),
            ("Vest Box", self.COLOR_VEST),
        ]

        # Calculate legend size
        box_w = 175
        item_h = 24
        header_h = 28
        total_h = header_h + len(legend_items) * item_h + 10

        # Draw semi-transparent background panel
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + total_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (x, y), (x + box_w, y + total_h), (80, 80, 80), 1)

        # Draw Header
        cv2.putText(
            img,
            "SafeEdge PPE Vision",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Draw legend rows
        cur_y = y + header_h + 10
        for label, color in legend_items:
            # Color indicator swatch
            cv2.rectangle(img, (x + 10, cur_y - 12), (x + 24, cur_y + 2), color, -1)
            cv2.rectangle(img, (x + 10, cur_y - 12), (x + 24, cur_y + 2), (255, 255, 255), 1)

            cv2.putText(
                img,
                label,
                (x + 32, cur_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cur_y += item_h

        return img

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Detection],
        assessments: List[WorkerAssessment],
    ) -> np.ndarray:
        """
        Draws bounding boxes, worker assessment banners, and legend on a copy of the image.

        Args:
            image: Original OpenCV image (NumPy array).
            detections: List of all detected objects.
            assessments: List of worker assessment results.

        Returns:
            Annotated OpenCV image as a NumPy array.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for visualization.")

        annotated = image.copy()
        img_h, img_w = annotated.shape[:2]

        # 1. Draw individual PPE item bounding boxes first (helmet, vest, no_ppe)
        for det in detections:
            if det.category == "person":
                continue  # Person boxes are handled via worker assessment for unified coloring

            color = self._get_item_color(det.category)
            x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.thickness)

            label = f"{det.class_name} {det.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.40, self.font_scale * 0.8)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Draw label tag
            tag_y1 = max(0, y1 - th - 4)
            tag_y2 = y1
            tag_x2 = min(img_w, x1 + tw + 6)
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 3),
                font,
                font_scale,
                self.COLOR_TEXT_BLACK,
                1,
                cv2.LINE_AA,
            )

        # 2. Draw worker bounding boxes and assessment banners
        for assessment in assessments:
            status = assessment.overall_status
            status_color = self._get_status_color(status)
            x1, y1, x2, y2 = [int(round(c)) for c in assessment.person_box]

            # Worker bounding box colored by compliance status
            cv2.rectangle(annotated, (x1, y1), (x2, y2), status_color, self.thickness + 1)

            # Status short badges
            h_sym = "H:OK" if assessment.helmet_status == PPEStatus.DETECTED else ("H:NO" if assessment.helmet_status == PPEStatus.MISSING else "H:?")
            v_sym = "V:OK" if assessment.vest_status == PPEStatus.DETECTED else ("V:NO" if assessment.vest_status == PPEStatus.MISSING else "V:?")

            banner_text = f"{assessment.worker_id.upper()}: {status.value} [{h_sym} {v_sym}]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(banner_text, font, self.font_scale, 2)

            banner_y1 = max(0, y1 - th - 8)
            banner_y2 = y1
            banner_x2 = min(img_w, x1 + tw + 10)

            # Banner background
            cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), status_color, -1)
            # Banner text
            cv2.putText(
                annotated,
                banner_text,
                (x1 + 5, banner_y2 - 5),
                font,
                self.font_scale,
                self.COLOR_TEXT_WHITE if status != WorkerPPEStatus.REVIEW_REQUIRED else self.COLOR_TEXT_BLACK,
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Legend if enabled
        if self.show_legend and img_w >= 300 and img_h >= 250:
            annotated = self.draw_legend(annotated, x=15, y=15)

        return annotated
