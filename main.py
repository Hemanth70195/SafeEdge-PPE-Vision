"""
SafeEdge PPE Vision - Main Command-Line Interface.
Phase 1: Construction Worker PPE Detection & Compliance Assessment Engine.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Adjust path for direct execution
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.model_loader import ModelLoader, ModelLoadError, IncompatibleModelError
from core.ppe_detector import PPEDetector
from core.ppe_assessor import PPEAssessor, WorkerPPEStatus, PPEStatus
from core.visualizer import PPEVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("SafeEdge")


def print_banner():
    banner = """
================================================================================
                    SafeEdge PPE Vision -- Phase 1 Engine                       
       Construction Personal Protective Equipment Compliance Monitoring         
================================================================================
"""
    print(banner)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="SafeEdge-PPE-Vision",
        description="Detect construction workers and verify PPE compliance (Hardhat & Safety Vest).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show help
  python main.py --help

  # Process an image with default verified model
  python main.py --image "data/site_workers.jpg"

  # Process with custom confidence and explicit output path
  python main.py --image "data/site.jpg" --conf 0.40 --output "output/reviewed_site.jpg"

  # Run on CPU explicitly
  python main.py --image "data/site.jpg" --device cpu
        """,
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to the input construction image file (.jpg, .jpeg, .png).",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to YOLO PPE model weights file (.pt). Defaults to configured models/best.pt.",
    )

    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=None,
        help="Confidence threshold for detections (0.0 to 1.0). Default from config: 0.35.",
    )

    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda", "gpu"],
        default="auto",
        help="Target compute device: 'auto' (default), 'cpu', or 'cuda'.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Destination path for annotated output image. Defaults to output/annotated_<filename>.",
    )

    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable status color legend on the output image.",
    )

    return parser.parse_args()


def process_image(
    image_path_str: str,
    model_path: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    device: str = "auto",
    output_path_str: Optional[str] = None,
    show_legend: bool = True,
) -> int:
    """
    Executes the end-to-end PPE detection and assessment pipeline on an image.
    """
    image_path = Path(image_path_str).resolve()
    if not image_path.exists():
        print(f"\n[ERROR] Input image not found: '{image_path}'", file=sys.stderr)
        return 1

    if not image_path.is_file():
        print(f"\n[ERROR] Provided path is not a file: '{image_path}'", file=sys.stderr)
        return 1

    # Load configuration
    config = ModelLoader.load_config()
    if not show_legend:
        config.setdefault("visualization", {})["show_legend"] = False

    # 1. Load actual YOLO PPE model
    print(f"\n[1/6] Loading PPE detection model...")
    try:
        detector = PPEDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            config=config,
        )
        print(f"      Model loaded successfully on device: {detector.model.device}")
        print(f"      Mapped PPE categories: {detector.category_mapping}")
    except IncompatibleModelError as e:
        print(f"\n[ERROR] Incompatible model detected:\n{e}", file=sys.stderr)
        return 1
    except ModelLoadError as e:
        print(f"\n[ERROR] Model loading failed:\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error loading model: {e}", file=sys.stderr)
        return 1

    # 2. Read and validate image
    print(f"[2/6] Reading image: '{image_path}'...")
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        print(f"\n[ERROR] Could not decode image file '{image_path}'. Ensure it is a valid JPEG/PNG.", file=sys.stderr)
        return 1

    img_h, img_w = image.shape[:2]
    print(f"      Image resolution: {img_w}x{img_h} pixels")

    # 3. Run PPE detection
    print(f"[3/6] Running inference (confidence threshold: {detector.conf_threshold})...")
    detections = detector.detect(image)
    print(f"      Total raw detections: {len(detections)}")

    # Group counts
    category_counts = {}
    for d in detections:
        category_counts[d.class_name] = category_counts.get(d.class_name, 0) + 1
    for cname, count in category_counts.items():
        print(f"        - {cname}: {count}")

    # 4. Assess PPE compliance
    print(f"[4/6] Performing spatial PPE association and compliance assessment...")
    assessor = PPEAssessor(config=config)
    assessments = assessor.assess(detections, image_shape=(img_h, img_w))
    print(f"      Workers evaluated: {len(assessments)}")

    # 5. Draw annotations
    print(f"[5/6] Generating annotated image...")
    visualizer = PPEVisualizer(config=config)
    annotated_image = visualizer.visualize(image, detections, assessments)

    # 6. Save output image
    print(f"[6/6] Saving annotated image...")
    if output_path_str:
        out_path = Path(output_path_str).resolve()
        if out_path.is_dir():
            out_path = out_path / f"annotated_{image_path.name}"
    else:
        out_dir = Path(config.get("output", {}).get("default_output_dir", "output")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{image_path.name}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_success = cv2.imwrite(str(out_path), annotated_image)
    if not save_success:
        print(f"\n[ERROR] Failed to save output image to '{out_path}'.", file=sys.stderr)
        return 1

    print(f"      Annotated image successfully saved to: {out_path}")

    # 7. Print summary report
    print("\n" + "=" * 80)
    print(f"                        PPE ASSESSMENT SUMMARY                          ")
    print("=" * 80)
    print(f" Source Image  : {image_path.name} ({img_w}x{img_h} px)")
    print(f" Output Image  : {out_path}")
    print(f" Workers Found : {len(assessments)}")
    print("-" * 80)

    if not assessments:
        print("  [INFO] No workers were detected in this image.")
    else:
        for a in assessments:
            stat_icon = "[OK]" if a.overall_status == WorkerPPEStatus.COMPLIANT else ("[VIOLATION]" if a.overall_status == WorkerPPEStatus.VIOLATION else "[REVIEW]")
            print(f"\n  Worker: {a.worker_id.upper()}  |  Overall Status: {stat_icon} {a.overall_status.value}")
            print(f"    Bounding Box   : {[int(c) for c in a.person_box]} (conf: {a.person_confidence:.2f})")
            print(f"    Safety Helmet  : {a.helmet_status.value}")
            print(f"    Safety Vest    : {a.vest_status.value}")
            print(f"    Explanation    : {a.explanation}")
            if a.confidence_scores:
                conf_str = ", ".join([f"{k}: {v:.2f}" for k, v in a.confidence_scores.items()])
                print(f"    Confidences    : {conf_str}")

    print("=" * 80 + "\n")
    return 0


def main():
    print_banner()
    args = parse_arguments()

    if not args.image:
        print("[INFO] No input image provided. Displaying help.")
        print("Use --image <path> to process an image. Run 'python main.py --help' for details.\n")
        return 0

    exit_code = process_image(
        image_path_str=args.image,
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        output_path_str=args.output,
        show_legend=not args.no_legend,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
