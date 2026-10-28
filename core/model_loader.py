"""
Model Loader for SafeEdge PPE Vision.
Handles verified YOLO PPE model weight validation, device resolution, and caching.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple

logger = logging.getLogger("SafeEdge.ModelLoader")


class ModelLoadError(Exception):
    """Raised when the model weights cannot be located or loaded."""
    pass


class IncompatibleModelError(Exception):
    """Raised when the loaded model does not support the required PPE detection classes."""
    pass


class ModelLoader:
    """
    Thread-safe / reusable loader for verified PPE YOLO models.
    Caches loaded models to avoid re-instantiation across inferences.
    """

    _cached_model = None
    _cached_model_path: Optional[str] = None
    _cached_device: Optional[str] = None
    _cached_class_mapping: Optional[Dict[str, int]] = None

    # Standard default synonyms for resolving model classes across different training setups
    DEFAULT_REQUIRED_CATEGORIES = {
        "person": ["person", "worker", "human"],
        "helmet": ["hardhat", "helmet", "safety_helmet", "hard_hat", "safety helmet"],
        "vest": ["safety vest", "vest", "safety_vest", "high_visibility_vest", "high-visibility vest", "reflective_vest"],
    }

    DEFAULT_OPTIONAL_CATEGORIES = {
        "no_helmet": ["no-hardhat", "no_hardhat", "no_helmet", "no-helmet", "no hardhat"],
        "no_vest": ["no-safety vest", "no-safety-vest", "no_vest", "no-vest", "no safety vest"],
        "mask": ["mask", "protective_mask"],
        "no_mask": ["no-mask", "no_mask"],
    }

    @classmethod
    def resolve_device(cls, requested_device: str = "auto") -> str:
        """
        Resolves device string to 'cuda' or 'cpu'.
        """
        requested = (requested_device or "auto").strip().lower()
        if requested in ["cuda", "gpu"]:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            except ImportError:
                return "cpu"
        elif requested == "cpu":
            return "cpu"
        else:
            # Auto-detection
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Loads project configuration from settings.json.
        """
        if config_path is None:
            # Try default config location relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "settings.json"

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse config file at {config_path}: {e}. Using defaults.")
        return {}

    @classmethod
    def validate_model_classes(
        cls,
        model_classes: Dict[int, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Validates that the model includes all required PPE categories (person, helmet, vest).
        Never permits a standard person-only COCO model to pass as a PPE detector.

        Returns:
            Dict mapping standard category names ('person', 'helmet', 'vest', etc.) to model class IDs.
        """
        if not model_classes:
            raise IncompatibleModelError("Model exposed an empty or invalid class names dictionary.")

        # Lowercase mapping for robust matching: {class_id: "hardhat"} -> {0: "hardhat"}
        id_to_norm = {cid: str(cname).strip().lower() for cid, cname in model_classes.items()}
        norm_to_id = {v: k for k, v in id_to_norm.items()}

        # Load category synonyms from config or use defaults
        class_mappings_cfg = (config or {}).get("class_mappings", {})
        required_targets = (config or {}).get("model", {}).get("required_ppe_types", ["helmet", "vest"])
        all_required = ["person"] + [t for t in required_targets if t != "person"]

        resolved_category_to_id: Dict[str, int] = {}
        missing_categories: List[str] = []

        for category in all_required:
            synonyms = class_mappings_cfg.get(category, cls.DEFAULT_REQUIRED_CATEGORIES.get(category, [category]))
            synonyms_norm = [s.strip().lower() for s in synonyms]

            matched_id = None
            for syn in synonyms_norm:
                if syn in norm_to_id:
                    matched_id = norm_to_id[syn]
                    break

            if matched_id is not None:
                resolved_category_to_id[category] = matched_id
            else:
                missing_categories.append(category)

        # Check optional categories (e.g. no_helmet, no_vest, mask)
        for category, default_syns in cls.DEFAULT_OPTIONAL_CATEGORIES.items():
            synonyms = class_mappings_cfg.get(category, default_syns)
            synonyms_norm = [s.strip().lower() for s in synonyms]
            for syn in synonyms_norm:
                if syn in norm_to_id:
                    resolved_category_to_id[category] = norm_to_id[syn]
                    break

        if missing_categories:
            available_classes = [f"{cid}: '{name}'" for cid, name in model_classes.items()]
            raise IncompatibleModelError(
                f"Incompatible YOLO model: Missing required PPE categories: {missing_categories}.\n"
                f"Available classes in this model ({len(available_classes)}): {', '.join(available_classes)}.\n"
                f"A generic model (e.g. COCO person-only) cannot be used as a PPE detector. "
                f"Please provide a genuine PPE-trained model weights file."
            )

        return resolved_category_to_id

    @classmethod
    def load_model(
        cls,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config_path: Optional[Path] = None,
        force_reload: bool = False,
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Loads and validates a YOLO model from disk.
        Returns (model_instance, resolved_category_mapping).
        """
        config = cls.load_config(config_path)

        # Resolve model path
        if not model_path:
            model_cfg_path = config.get("model", {}).get("model_path", "models/best.pt")
            base_dir = Path(__file__).resolve().parent.parent
            resolved_path = (base_dir / model_cfg_path).resolve()
        else:
            resolved_path = Path(model_path).resolve()

        resolved_device = cls.resolve_device(device or config.get("model", {}).get("device", "auto"))

        # Check cache
        if (
            not force_reload
            and cls._cached_model is not None
            and cls._cached_model_path == str(resolved_path)
            and cls._cached_device == resolved_device
            and cls._cached_class_mapping is not None
        ):
            return cls._cached_model, cls._cached_class_mapping

        # Validate file existence
        if not resolved_path.exists():
            raise ModelLoadError(
                f"Model weights file not found at: '{resolved_path}'.\n"
                f"Please ensure the verified PPE model weights are downloaded to this path.\n"
                f"To download the verified model automatically, run:\n"
                f"  python models/download_model.py"
            )

        if resolved_path.stat().st_size == 0:
            raise ModelLoadError(
                f"Model weights file at '{resolved_path}' is empty (0 bytes). Please re-download."
            )

        # Import Ultralytics YOLO safely
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(
                f"Ultralytics package is not installed or import failed: {e}. "
                f"Run: pip install -r requirements.txt"
            ) from e

        # Load model weights
        try:
            model = YOLO(str(resolved_path))
            model.to(resolved_device)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YOLO model from '{resolved_path}': {e}"
            ) from e

        # Extract and validate class names
        raw_classes = getattr(model, "names", None)
        if raw_classes is None and hasattr(model, "model") and hasattr(model.model, "names"):
            raw_classes = model.model.names

        if isinstance(raw_classes, (list, tuple)):
            raw_classes = {i: name for i, name in enumerate(raw_classes)}

        resolved_category_mapping = cls.validate_model_classes(raw_classes, config=config)

        # Cache instance
        cls._cached_model = model
        cls._cached_model_path = str(resolved_path)
        cls._cached_device = resolved_device
        cls._cached_class_mapping = resolved_category_mapping

        logger.info(
            f"Successfully loaded PPE model from {resolved_path} on {resolved_device}. "
            f"Mapped categories: {resolved_category_mapping}"
        )
        return model, resolved_category_mapping

    @classmethod
    def clear_cache(cls):
        """Clears the cached model instance."""
        cls._cached_model = None
        cls._cached_model_path = None
        cls._cached_device = None
        cls._cached_class_mapping = None
