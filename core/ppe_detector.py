"""
PPE Object Detector for SafeEdge PPE Vision.
Performs YOLO-based inference on OpenCV images to detect workers and PPE items.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .model_loader import ModelLoader

logger = logging.getLogger("SafeEdge.PPEDetector")


@dataclass
class Detection:
    """
    Structured representation of a single detected object.
    """
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    category: str      # Normalized category: 'person', 'helmet', 'vest', 'no_helmet', 'no_vest', 'mask', 'other'
    class_id: int

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "category": self.category,
            "class_id": self.class_id,
        }


class PPEDetector:
    """
    Reusable PPE object detector.
    Accepts OpenCV images (NumPy arrays) and outputs structured Detection instances.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        device: Optional[str] = None,
        category_mapping: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the PPE detector.
        If a model instance is provided directly, it is used (enables mocking in tests).
        Otherwise, loads and validates the verified model via ModelLoader.
        """
        self.config = config or ModelLoader.load_config()

        # Resolve confidence threshold
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        else:
            self.conf_threshold = float(self.config.get("model", {}).get("confidence_threshold", 0.35))

        if model is not None:
            self.model = model
            if category_mapping is not None:
                self.category_mapping = category_mapping
            else:
                names = getattr(model, "names", {})
                if isinstance(names, (list, tuple)):
                    names = {i: n for i, n in enumerate(names)}
                self.category_mapping = ModelLoader.validate_model_classes(names, self.config)
        else:
            self.model, self.category_mapping = ModelLoader.load_model(
                model_path=model_path,
                device=device,
            )

        # Invert category mapping for rapid category lookup by class_id
        # {category: id} -> {id: category}
        self.id_to_category: Dict[int, str] = {cid: cat for cat, cid in self.category_mapping.items()}

    def _resolve_category(self, class_id: int, class_name: str) -> str:
        """
        Resolves category for a given detection by class ID and class name.
        """
        if class_id in self.id_to_category:
            return self.id_to_category[class_id]

        cname_lower = class_name.strip().lower()
        class_mappings_cfg = self.config.get("class_mappings", {})
        for cat, syns in class_mappings_cfg.items():
            for syn in syns:
                if syn.strip().lower() == cname_lower:
                    return cat

        return "other"

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Performs PPE and person detection on an input OpenCV image.

        Args:
            image: BGR or RGB image as a NumPy array (H x W x C or H x W).
            conf_threshold: Optional override for the detection confidence threshold.

        Returns:
            List of structured Detection objects.

        Raises:
            ValueError: If the input image is None, not a NumPy array, or has invalid dimensions.
        """
        # Validate image input
        if image is None:
            raise ValueError("Input image is None. Expected a valid NumPy image array.")

        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input image must be a numpy.ndarray, got {type(image).__name__}.")

        if image.size == 0 or image.ndim < 2:
            raise ValueError(
                f"Input image has invalid shape {getattr(image, 'shape', None)}. "
                f"Image must be non-empty with at least 2 dimensions (H x W)."
            )

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Image has non-positive dimensions: height={h}, width={w}.")

        conf = float(conf_threshold if conf_threshold is not None else self.conf_threshold)

        # Run inference using the preloaded YOLO model
        try:
            results = self.model(image, conf=conf, verbose=False)
        except Exception as e:
            logger.error(f"Inference error encountered: {e}")
            raise RuntimeError(f"YOLO model inference failed: {e}") from e

        if not results:
            return []

        detections: List[Detection] = []
        result = results[0]

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # Get class names dictionary from model
        names_dict = getattr(self.model, "names", {})
        if isinstance(names_dict, (list, tuple)):
            names_dict = {i: n for i, n in enumerate(names_dict)}

        # Extract detections
        for box in boxes:
            box_conf = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])
            if box_conf < conf:
                continue

            class_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
            raw_class_name = names_dict.get(class_id, f"class_{class_id}")
            category = self._resolve_category(class_id, raw_class_name)

            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
            x1, y1, x2, y2 = [float(coord) for coord in xyxy]

            # Clip coordinates to image boundary
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detection = Detection(
                class_name=raw_class_name,
                confidence=box_conf,
                bbox=[x1, y1, x2, y2],
                category=category,
                class_id=class_id,
            )
            detections.append(detection)

        return detections

    @staticmethod
    def filter_by_category(detections: List[Detection], category: str) -> List[Detection]:
        """Filters detections list by normalized category."""
        return [d for d in detections if d.category == category]
