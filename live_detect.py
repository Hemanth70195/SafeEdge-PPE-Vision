"""
SafeEdge PPE Vision — Live Video Detector
==========================================
Simulates real-time CCTV PPE monitoring.

Usage:
    python live_detect.py                            # auto-picks first video in data/
    python live_detect.py --video data/myvideo.mp4  # pick a specific video
    python live_detect.py --video data/myvideo.mp4 --conf 0.35

Controls (while window is open):
    SPACE    -- pause / resume
    Q / ESC  -- quit
    S        -- save current frame as screenshot

Output video is saved to output/live_<original_name>.mp4 automatically.
"""

import sys
import os
import argparse
import time
import glob
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.ppe_detector import PPEDetector
from core.ppe_assessor import PPEAssessor, WorkerPPEStatus, PPEStatus
from core.visualizer import PPEVisualizer
from core.tracker import SimpleTracker

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def find_video():
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = glob.glob(str(ROOT / "data" / ext))
        if matches:
            return matches[0]
    return ""


def draw_hud(frame, frame_idx, fps, total_workers, violations, compliant, paused, loop_count=1):
    h, w = frame.shape[:2]
    panel_w, panel_h = 270, 162
    px, py = w - panel_w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cy = py + 22

    cv2.putText(frame, "SafeEdge LIVE Monitor", (px + 8, cy), font, 0.42, (200, 220, 255), 1, cv2.LINE_AA)
    cy += 22
    fps_color = (100, 255, 100) if fps >= 10 else (0, 165, 255)
    loop_badge = f"Loop #{loop_count}" if loop_count > 1 else "Loop: ON"
    cv2.putText(frame, f"FPS: {fps:.1f}   Frame: {frame_idx}  ({loop_badge})", (px + 8, cy), font, 0.38, fps_color, 1, cv2.LINE_AA)
    cy += 20
    cv2.line(frame, (px + 8, cy), (px + panel_w - 8, cy), (60, 60, 60), 1)
    cy += 14
    cv2.putText(frame, f"Workers in frame:  {total_workers}", (px + 8, cy), font, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    cy += 18
    cv2.putText(frame, f"Compliant :        {compliant}", (px + 8, cy), font, 0.40, (50, 205, 50), 1, cv2.LINE_AA)
    cy += 18
    cv2.putText(frame, f"Violations:        {violations}", (px + 8, cy), font, 0.40, (60, 60, 220), 1, cv2.LINE_AA)
    cy += 22
    status_text = "  PAUSED  " if paused else "  LIVE (LOOP)  "
    status_color = (0, 165, 255) if paused else (50, 205, 50)
    (tw, th), _ = cv2.getTextSize(status_text, font, 0.42, 2)
    sx = px + (panel_w - tw) // 2
    cv2.rectangle(frame, (sx - 4, cy - th - 2), (sx + tw + 4, cy + 4), status_color, -1)
    cv2.putText(frame, status_text, (sx, cy), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_controls_bar(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 22), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.putText(frame, "  SPACE: Pause/Resume    Q / ESC: Exit    S: Screenshot    [Loop: Active]",
                (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    return frame


def run(video_path, conf=0.35, save=True, stride=1, loop=True):
    print(f"\n{BOLD}{CYAN}SafeEdge PPE Vision — Live Detector{RESET}")
    print(f"{CYAN}Video : {video_path}{RESET}")
    print(f"{CYAN}Conf  : {conf}   Stride: {stride}{RESET}")
    print(f"{CYAN}Loop  : {'ON (continuous loop)' if loop else 'OFF (play once)'}{RESET}")
    print("Controls: SPACE=pause  Q/ESC=exit  S=screenshot\n")

    if not os.path.isfile(video_path):
        print(f"{RED}ERROR: Video not found: {video_path}{RESET}")
        sys.exit(1)

    print("Loading PPE model...", end=" ", flush=True)
    detector   = PPEDetector(conf_threshold=conf)
    assessor   = PPEAssessor()
    visualizer = PPEVisualizer()
    tracker    = SimpleTracker(iou_threshold=0.30, max_lost_frames=8)
    print(f"{GREEN}OK{RESET}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{RED}ERROR: Cannot open video.{RESET}")
        sys.exit(1)

    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_w}x{vid_h} @ {vid_fps:.1f}fps  |  {total_frames} frames\n")

    writer   = None
    out_path = ""
    if save:
        out_dir  = ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        stem     = Path(video_path).stem
        out_path = str(out_dir / f"live_{stem}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, vid_fps, (vid_w, vid_h))
        print(f"Saving → {out_path}")

    win_name = "SafeEdge PPE — Live Detection  (Q to exit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(vid_w, 1280), min(vid_h, 720))

    frame_idx    = 0
    loop_count   = 1
    paused       = False
    fps_display  = 0.0
    t_prev       = time.time()
    annotated    = None
    assessments  = []
    violations   = 0
    compliant    = 0
    screenshot_dir = ROOT / "output" / "screenshots"

    while True:
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("s"), ord("S")) and annotated is not None:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sc = str(screenshot_dir / f"screenshot_{ts}.jpg")
            cv2.imwrite(sc, annotated)
            print(f"{GREEN}Screenshot: {sc}{RESET}")

        if paused:
            if annotated is not None:
                disp = draw_hud(annotated.copy(), frame_idx, fps_display,
                                len(assessments), violations, compliant, paused=True,
                                loop_count=loop_count)
                disp = draw_controls_bar(disp)
                cv2.imshow(win_name, disp)
            continue

        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"\n{YELLOW}End of video (unable to loop).{RESET}")
                        break
                loop_count += 1
                frame_idx = 0
                tracker = SimpleTracker(iou_threshold=0.30, max_lost_frames=8)
                if writer is not None:
                    writer.release()
                    writer = None
                    if save and out_path:
                        print(f"\n{GREEN}First pass saved → {out_path}{RESET}")
                print(f"\n{YELLOW}↺ Looping video (Loop #{loop_count}) | Press Q or ESC to exit anytime{RESET}")
            else:
                print(f"\n{YELLOW}End of video.{RESET}")
                break
        frame_idx += 1

        if stride > 1 and (frame_idx % stride != 0):
            continue

        detections  = detector.detect(frame)
        assessments = assessor.assess(detections, image_shape=frame.shape[:2])

        assessments = tracker.update(assessments, frame_idx)

        violations = sum(1 for a in assessments if a.overall_status == WorkerPPEStatus.VIOLATION)
        compliant  = sum(1 for a in assessments if a.overall_status == WorkerPPEStatus.COMPLIANT)

        annotated = visualizer.visualize(frame, detections, assessments)

        t_now       = time.time()
        elapsed     = t_now - t_prev
        fps_display = 1.0 / elapsed if elapsed > 0 else 0.0
        t_prev      = t_now

        annotated = draw_hud(annotated, frame_idx, fps_display,
                             len(assessments), violations, compliant, paused=False,
                             loop_count=loop_count)
        annotated = draw_controls_bar(annotated)

        if writer is not None:
            writer.write(annotated)

        cv2.imshow(win_name, annotated)

        if frame_idx % 30 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            loop_tag = f" [Loop #{loop_count}]" if loop_count > 1 else ""
            print(f"  Frame {frame_idx:>5}/{total_frames} ({pct:.0f}%){loop_tag} | "
                  f"Workers:{len(assessments)}  "
                  f"{GREEN}OK:{compliant}{RESET}  "
                  f"{RED}VIOL:{violations}{RESET}  "
                  f"FPS:{fps_display:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{BOLD}{GREEN}Done! Detector closed.{RESET}")
    if save and out_path and os.path.isfile(out_path):
        print(f"Annotated video saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeEdge PPE Live Detector")
    parser.add_argument("--video",    default="",         help="Path to video (default: first file in data/)")
    parser.add_argument("--conf",     type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--stride",   type=int,   default=1,    help="Process every Nth frame (default: 1)")
    parser.add_argument("--no-save",  action="store_true",      help="Disable saving output video")
    parser.add_argument("--no-loop",  action="store_true",      help="Disable loop (play once and exit)")
    args = parser.parse_args()

    video = args.video or find_video()
    if not video:
        print(f"{RED}No video found in data/. Pass --video path/to/video.mp4{RESET}")
        sys.exit(1)

    run(video_path=video, conf=args.conf, save=not args.no_save, stride=args.stride, loop=not args.no_loop)
