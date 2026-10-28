"""
Model Downloader Utility for SafeEdge PPE Vision.
Downloads verified pretrained YOLO PPE model weights from Hugging Face Hub.
"""

import os
import sys
import argparse
import requests
from pathlib import Path

DEFAULT_URL = "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt"
DEFAULT_TARGET_PATH = Path(__file__).resolve().parent / "best.pt"
MIN_VALID_SIZE_BYTES = 5_000_000  # Expected ~6.25MB


def download_model(url: str = DEFAULT_URL, dest_path: Path = DEFAULT_TARGET_PATH, force: bool = False) -> Path:
    """
    Downloads model weights from a remote URL to local disk.

    Args:
        url: Remote URL of the model weights (.pt).
        dest_path: Destination local file path.
        force: If True, overwrite existing file.

    Returns:
        Path to the downloaded model file.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        file_size = dest_path.stat().st_size
        if file_size >= MIN_VALID_SIZE_BYTES:
            print(f"[INFO] Verified model weights already exist at: {dest_path} ({file_size / 1_000_000:.2f} MB)")
            return dest_path
        else:
            print(f"[WARN] Existing model file is undersized ({file_size} bytes). Re-downloading...")

    print(f"[INFO] Downloading verified PPE detection model from:\n  {url}")
    print(f"[INFO] Destination: {dest_path}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SafeEdge-PPE-Vision/1.0"}
    response = requests.get(url, stream=True, headers=headers, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download model weights. HTTP status code {response.status_code}. "
            f"URL: {url}"
        )

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 64 * 1024
    downloaded = 0

    temp_path = dest_path.with_suffix(".tmp")
    try:
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r[INFO] Progress: {downloaded / 1_000_000:.2f} / {total_size / 1_000_000:.2f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r[INFO] Downloaded {downloaded / 1_000_000:.2f} MB", end="", flush=True)

        print("\n[INFO] Download complete. Validating integrity...")

        temp_size = temp_path.stat().st_size
        if temp_size < MIN_VALID_SIZE_BYTES:
            raise RuntimeError(
                f"Downloaded model file is incomplete or corrupted ({temp_size} bytes, expected >= {MIN_VALID_SIZE_BYTES} bytes)."
            )

        # Atomic replace
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        print(f"[SUCCESS] Model successfully verified and saved to: {dest_path}")
        return dest_path

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Error during model download: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Download verified PPE detection model weights.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Model download URL")
    parser.add_argument("--dest", default=str(DEFAULT_TARGET_PATH), help="Target local path")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    args = parser.parse_args()

    try:
        download_model(url=args.url, dest_path=Path(args.dest), force=args.force)
    except Exception as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
