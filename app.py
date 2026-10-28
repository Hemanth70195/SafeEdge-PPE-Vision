"""
SafeEdge AI — Construction Safety Monitoring System.
Phase 5: Unified Platform Integrating Computer Vision PPE Compliance & Environmental Hazard Simulation.

Sections:
1. Executive Safety Overview: Unified cross-system intelligence dashboard
2. PPE Image Detection: High-precision single-frame worker PPE compliance verification
3. PPE Video Detection: Multi-frame temporal worker tracking, video processing, and audit analytics
4. Environmental Hazard Simulator: Five-parameter real-time construction site hazard cockpit
5. Hazard History & Reports: Historical trend charts, transition logs, summary stats, and CSV/JSON downloads
"""

import os
import sys
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_loader import ModelLoader, ModelLoadError, IncompatibleModelError
from core.ppe_detector import PPEDetector, Detection
from core.ppe_assessor import PPEAssessor, WorkerAssessment, PPEStatus, WorkerPPEStatus
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
from core.environmental_simulator import (
    EnvironmentalSimulator,
    HazardStatus,
    DEFAULT_CONFIG_PATH,
    SIMULATED_DATA_DISCLAIMER_SHORT,
)
from core.environmental_dashboard import (
    get_or_create_simulator,
    render_dashboard_styles,
    render_simulator_view,
    render_history_reports_view,
    render_environmental_dashboard,
    PHASE3_DISCLAIMER_TITLE,
    PHASE3_DISCLAIMER_SUB,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeEdge AI - Construction Safety Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Custom Styling & Design Tokens
# -----------------------------------------------------------------------------
def inject_custom_styles():
    """Injects high-contrast, professional industrial safety CSS."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
        }
        .brand-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border-radius: 12px;
            padding: 22px 28px;
            color: #FFFFFF;
            margin-bottom: 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.12);
            border-left: 6px solid #38BDF8;
        }
        .brand-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-subtitle {
            font-size: 0.98rem;
            color: #94A3B8;
            margin-top: 6px;
            margin-bottom: 0;
            line-height: 1.4;
        }
        .dual-disclaimer-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 18px;
        }
        .disclaimer-card-vision {
            background-color: #EFF6FF;
            border-left: 5px solid #3B82F6;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #1E3A8A;
            line-height: 1.35;
        }
        .disclaimer-card-sim {
            background-color: #FEF3C7;
            border-left: 5px solid #F59E0B;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #92400E;
            line-height: 1.35;
        }
        .overview-status-card {
            background-color: #FFFFFF;
            border: 2px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            height: 100%;
        }
        .overview-card-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 6px;
        }
        .badge-compliant {
            background-color: #DCFCE7;
            color: #166534;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.82rem;
            display: inline-block;
        }
        .badge-violation {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.82rem;
            display: inline-block;
        }
        .badge-review {
            background-color: #FFEDD5;
            color: #9A3412;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.82rem;
            display: inline-block;
        }
        .action-card-link {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }
        .action-card-link:hover {
            border-color: #38BDF8;
            background-color: #F0F9FF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading verified YOLO PPE model into memory...")
def get_cached_detector(model_path: str, device: str) -> Tuple[PPEDetector, Dict[str, Any]]:
    """
    Caches the YOLO PPE detector instance in Streamlit memory so weights load only once.
    """
    config = ModelLoader.load_config()
    detector = PPEDetector(model_path=model_path, device=device, config=config)
    return detector, config


# -----------------------------------------------------------------------------
# Module A: Executive Safety Overview
# -----------------------------------------------------------------------------
def render_overview_module(sim: EnvironmentalSimulator):
    """
    Renders the unified executive overview dashboard connecting PPE compliance
    intelligence with environmental hazard telemetry.
    """
    st.markdown(
        """
        <div class="brand-banner">
            <h1 class="brand-title">🛡️ SafeEdge AI — Construction Safety Monitoring System</h1>
            <p class="brand-subtitle">
                Executive Safety Intelligence Dashboard. Integrating deep-learning computer vision for worker PPE compliance
                with real-time multi-hazard environmental risk simulation and audit reporting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent Cross-System Disclaimers
    st.markdown(
        f"""
        <div class="dual-disclaimer-grid">
            <div class="disclaimer-card-vision">
                <strong>🦺 Vision AI Engine (Live Active Inference):</strong>
                Worker detections and PPE compliance assessments are computed in real time using the verified deep learning
                model <code>Hansung-Cho/yolov8-ppe-detection</code> on uploaded construction site imagery and video.
            </div>
            <div class="disclaimer-card-sim">
                <strong>⚠️ Environmental Hazard Engine ({PHASE3_DISCLAIMER_TITLE}):</strong>
                Environmental parameters (Temperature, Humidity, Dust, Noise, Gas) represent software-simulated telemetry
                for risk modeling and procedure training. They do NOT reflect physical IoT sensor hardware.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Environmental Assessment
    env_site_ass = sim.assess_current_hazards()

    # Retrieve last analyzed PPE results if present in session
    last_img = st.session_state.get("last_image_results", None)
    last_vid = st.session_state.get("last_video_results", None)

    # 4 Executive KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="overview-status-card">
                <div class="overview-card-title">🤖 AI Vision Model Pipeline</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin: 4px 0;">YOLOv8n-PPE</div>
                <div style="font-size: 0.82rem; color: #475569;">Hansung-Cho / MIT License</div>
                <div style="margin-top: 8px;"><span class="badge-compliant">🟢 MODEL LOADED & VERIFIED</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if last_vid is not None:
            stats: VideoProcessingStats = last_vid["stats"]
            total_obs = max(1, stats.total_person_observations)
            comp_rate = (stats.compliant_observations / total_obs) * 100
            badge = '<span class="badge-compliant">🟢 COMPLIANT SITE</span>' if comp_rate >= 80 else '<span class="badge-violation">🔴 VIOLATIONS ACTIVE</span>'
            st.markdown(
                f"""
                <div class="overview-status-card">
                    <div class="overview-card-title">🦺 Site PPE Compliance</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin: 4px 0;">{comp_rate:.1f}% Compliant</div>
                    <div style="font-size: 0.82rem; color: #475569;">Tracked: {stats.unique_tracked_workers} workers | {stats.violation_observations} violations</div>
                    <div style="margin-top: 8px;">{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif last_img is not None:
            ass_list = last_img.get("assessments", [])
            total_w = len(ass_list)
            viol_w = sum(1 for a in ass_list if a.overall_status == WorkerPPEStatus.VIOLATION)
            comp_w = sum(1 for a in ass_list if a.overall_status == WorkerPPEStatus.COMPLIANT)
            comp_rate = (comp_w / max(1, total_w)) * 100
            badge = '<span class="badge-compliant">🟢 ALL COMPLIANT</span>' if viol_w == 0 else '<span class="badge-violation">🔴 VIOLATIONS DETECTED</span>'
            st.markdown(
                f"""
                <div class="overview-status-card">
                    <div class="overview-card-title">🦺 Site PPE Compliance</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin: 4px 0;">{comp_w}/{total_w} Compliant</div>
                    <div style="font-size: 0.82rem; color: #475569;">{viol_w} worker(s) missing required PPE</div>
                    <div style="margin-top: 8px;">{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="overview-status-card">
                    <div class="overview-card-title">🦺 Site PPE Compliance</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #64748B; margin: 4px 0;">Awaiting Inspection</div>
                    <div style="font-size: 0.82rem; color: #64748B;">Upload image or video in PPE modules</div>
                    <div style="margin-top: 8px;"><span class="badge-review">STANDBY FOR MEDIA</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col3:
        if env_site_ass.overall_status == HazardStatus.SAFE:
            status_html = '<span class="badge-compliant">🟢 NORMAL & SAFE</span>'
        elif env_site_ass.overall_status == HazardStatus.WARNING:
            status_html = '<span class="badge-review">🟠 ELEVATED WARNING</span>'
        else:
            status_html = '<span class="badge-violation">🔴 CRITICAL HAZARD</span>'

        st.markdown(
            f"""
            <div class="overview-status-card">
                <div class="overview-card-title">🏗️ Environmental Status (Sim)</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin: 4px 0;">{env_site_ass.overall_status.name}</div>
                <div style="font-size: 0.82rem; color: #475569;">Mode: {st.session_state.get('env_sim_mode', 'manual').upper()} | Ticks: #{sim.tick_count}</div>
                <div style="margin-top: 8px;">{status_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="overview-status-card">
                <div class="overview-card-title">🚨 Active Hazard Directives</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin: 4px 0;">{env_site_ass.danger_count} Danger / {env_site_ass.warning_count} Warn</div>
                <div style="font-size: 0.82rem; color: #475569;">Safe Parameters: {len(env_site_ass.safe_parameters)}/5</div>
                <div style="margin-top: 8px;"><span style="font-size: 0.78rem; font-weight: 700; color: #64748B;">SIMULATION BUFFER</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Cross-System Intelligence Feeds
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("🦺 Recent PPE Violations & Inspection Summary")
        if last_vid is not None:
            stats = last_vid["stats"]
            if stats.events:
                df_ev = pd.DataFrame(stats.events)
                viol_only = df_ev[df_ev["overall_status"] == "VIOLATION"]
                if not viol_only.empty:
                    st.dataframe(
                        viol_only[["timestamp_str", "worker_id", "helmet_status", "vest_status", "explanation"]].head(6),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("✅ **Zero Violations in Processed Video:** All detected workers fully equipped with hardhats and safety vests.")
            else:
                st.info("No frame-level events documented in current video session.")
        elif last_img is not None:
            ass_list = last_img.get("assessments", [])
            viols = [a for a in ass_list if a.overall_status == WorkerPPEStatus.VIOLATION]
            if viols:
                v_rows = [{
                    "Worker": a.worker_id.upper(),
                    "Status": a.overall_status.value,
                    "Helmet": a.helmet_status.value,
                    "Vest": a.vest_status.value,
                    "Explanation": a.explanation,
                } for a in viols]
                st.dataframe(pd.DataFrame(v_rows), use_container_width=True, hide_index=True)
            else:
                st.success("✅ **All Workers Fully Compliant:** Hardhats and high-visibility safety vests verified.")
        else:
            st.info("No PPE inspection media has been evaluated yet. Navigate to **PPE Image Detection** or **PPE Video Detection** to inspect construction personnel.")

    with c_right:
        st.subheader("🏗️ Recent Environmental Hazard Events (Simulated)")
        df_events = sim.get_hazard_events_df()
        if not df_events.empty:
            st.dataframe(df_events.head(6), use_container_width=True, hide_index=True)
        else:
            st.success("✅ **Baseline Environmental Conditions:** All 5 parameters operating within configured safe limits.")

    st.divider()

    # Quick Launch Navigation Cards
    st.subheader("🚀 Quick System Navigation")
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.markdown(
            """
            <div class="action-card-link">
                <div style="font-weight: 800; font-size: 1.05rem; color: #0F172A; margin-bottom: 4px;">📷 PPE Image Detection</div>
                <div style="font-size: 0.84rem; color: #64748B; margin-bottom: 8px;">Upload site photos or load demo samples for instant multi-worker compliance assessment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Image Inspector", key="btn_goto_img", use_container_width=True):
            st.session_state["safeedge_main_module_nav"] = "📷 PPE Image Detection"
            st.rerun()

    with q2:
        st.markdown(
            """
            <div class="action-card-link">
                <div style="font-weight: 800; font-size: 1.05rem; color: #0F172A; margin-bottom: 4px;">🎬 PPE Video Detection</div>
                <div style="font-size: 0.84rem; color: #64748B; margin-bottom: 8px;">Process construction footage with frame sampling, temporal tracking, and audit exports.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Video Inspector", key="btn_goto_vid", use_container_width=True):
            st.session_state["safeedge_main_module_nav"] = "🎬 PPE Video Detection"
            st.rerun()

    with q3:
        st.markdown(
            """
            <div class="action-card-link">
                <div style="font-weight: 800; font-size: 1.05rem; color: #0F172A; margin-bottom: 4px;">🏗️ Hazard Simulator</div>
                <div style="font-size: 0.84rem; color: #64748B; margin-bottom: 8px;">Simulate microclimate, dust, noise, and gas risks with manual sliders and auto drift.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Hazard Simulator", key="btn_goto_sim", use_container_width=True):
            st.session_state["safeedge_main_module_nav"] = "🏗️ Environmental Hazard Simulator"
            st.rerun()

    with q4:
        st.markdown(
            """
            <div class="action-card-link">
                <div style="font-weight: 800; font-size: 1.05rem; color: #0F172A; margin-bottom: 4px;">📊 History & Reports</div>
                <div style="font-size: 0.84rem; color: #64748B; margin-bottom: 8px;">Inspect time-series telemetry, transition event timelines, summary stats, and CSV/JSON downloads.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Reports & History", key="btn_goto_rep", use_container_width=True):
            st.session_state["safeedge_main_module_nav"] = "📊 Hazard History & Reports"
            st.rerun()


# -----------------------------------------------------------------------------
# Module B: PPE Image Detection
# -----------------------------------------------------------------------------
def render_image_detection_module(config: Dict[str, Any], resolved_model_path: str, device_option: str, conf_threshold: float):
    """
    Renders Section A of Phase 5: Complete single-image worker PPE detection,
    spatial compliance assessment, annotated preview, and tabular report.
    """
    st.markdown('<div class="main-header">📷 PPE Image Detection & Compliance Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">High-Precision Computer Vision Inspection for Construction Personnel Hardhat and Safety Vest Compliance</div>',
        unsafe_allow_html=True,
    )

    # Input Controls Bar
    in_col1, in_col2 = st.columns([3, 1])

    with in_col1:
        uploaded_image = st.file_uploader(
            "Upload Construction Site Image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a standard high-resolution site photograph (.jpg, .png, .webp).",
            key="ppe_image_uploader",
        )

    with in_col2:
        st.markdown("**Demonstration Preset:**")
        sample_img_path = PROJECT_ROOT / "data" / "sample_workers.jpg"
        load_sample = st.button("🖼️ Load Demo Sample Image", use_container_width=True, key="btn_load_demo_image")

    # Image source resolution
    image_to_process = None
    source_name = ""

    if load_sample and sample_img_path.exists():
        image_to_process = cv2.imread(str(sample_img_path))
        source_name = "sample_workers.jpg (Built-in Demo)"
        st.session_state["ppe_demo_image_active"] = True

    elif uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_name = uploaded_image.name
        st.session_state["ppe_demo_image_active"] = False

    elif st.session_state.get("ppe_demo_image_active", False) and sample_img_path.exists():
        image_to_process = cv2.imread(str(sample_img_path))
        source_name = "sample_workers.jpg (Built-in Demo)"

    if image_to_process is None:
        st.info("👋 Upload a site image above or click **Load Demo Sample Image** to run the PPE inspection pipeline.")
        return

    # Process Image
    img_h, img_w = image_to_process.shape[:2]

    with st.spinner("Executing YOLO PPE model inference and spatial compliance analysis..."):
        try:
            detector, _ = get_cached_detector(resolved_model_path, device_option)
        except Exception as e:
            st.error(f"❌ Failed to load verified PPE detection model: {e}")
            return

        detections = detector.detect(image_to_process, conf_threshold=conf_threshold)
        assessor = PPEAssessor(config=config)
        assessments = assessor.assess(detections, image_shape=(img_h, img_w))
        visualizer = PPEVisualizer(config=config)
        annotated_image = visualizer.visualize(image_to_process, detections, assessments)

        # Store in session state for cross-system executive overview
        st.session_state["last_image_results"] = {
            "source_name": source_name,
            "assessments": assessments,
            "detections": detections,
            "image_shape": (img_h, img_w),
        }

    # Summary Metrics Row
    total_workers = len(assessments)
    compliant_w = sum(1 for a in assessments if a.overall_status == WorkerPPEStatus.COMPLIANT)
    violation_w = sum(1 for a in assessments if a.overall_status == WorkerPPEStatus.VIOLATION)
    review_w = sum(1 for a in assessments if a.overall_status == WorkerPPEStatus.REVIEW_REQUIRED)

    st.markdown("##### 🛡️ Inspection Compliance Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Workers Evaluated", total_workers)
    with kpi2:
        st.metric("Compliant Personnel", compliant_w, delta=f"{compliant_w}/{total_workers} OK", delta_color="normal")
    with kpi3:
        st.metric("Confirmed Violations", violation_w, delta=f"{violation_w} VIOLATIONS", delta_color="inverse")
    with kpi4:
        st.metric("Review Required", review_w, delta=f"{review_w} REVIEW", delta_color="off")

    st.divider()

    # Visual Inspection Tabs
    tab_anno, tab_orig = st.tabs(["🖼️ Annotated Compliance Inspection View", "📷 Original Source Image"])

    with tab_anno:
        # Convert BGR to RGB for Streamlit rendering
        rgb_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_annotated, caption=f"Annotated Inspection: {source_name} ({img_w}×{img_h} px)", use_container_width=True)

    with tab_orig:
        rgb_original = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
        st.image(rgb_original, caption=f"Original Unprocessed Source: {source_name}", use_container_width=True)

    st.divider()

    # Worker Compliance Breakdown Table
    st.subheader("Worker Compliance Assessment Breakdown")

    if assessments:
        table_rows = []
        for a in assessments:
            if a.overall_status == WorkerPPEStatus.COMPLIANT:
                stat_badge = "🟢 COMPLIANT"
            elif a.overall_status == WorkerPPEStatus.VIOLATION:
                stat_badge = "🔴 VIOLATION"
            else:
                stat_badge = "🟠 REVIEW_REQUIRED"

            table_rows.append({
                "Worker ID": a.worker_id.upper(),
                "Overall Compliance": stat_badge,
                "Hardhat (Safety Helmet)": a.helmet_status.value,
                "Safety Vest (High-Vis)": a.vest_status.value,
                "Detection Confidence": f"{a.person_confidence:.2f}",
                "Bounding Box [x1, y1, x2, y2]": f"{[int(c) for c in a.person_box]}",
                "Explanation": a.explanation,
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No construction workers were detected in this image. Ensure personnel are visible and adjust confidence threshold if necessary.")

    # Detected Classes Summary
    st.markdown("##### Detected PPE Objects Count")
    class_counts: Dict[str, int] = {}
    for d in detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

    c_cols = st.columns(len(class_counts) if class_counts else 1)
    for idx, (cname, cval) in enumerate(class_counts.items()):
        with c_cols[idx]:
            st.metric(cname, cval)

    # Download Output Image
    st.divider()
    success, encoded_jpg = cv2.imencode(".jpg", annotated_image)
    if success:
        st.download_button(
            label="📥 Download Annotated Inspection Image (JPEG)",
            data=encoded_jpg.tobytes(),
            file_name=f"safeedge_inspection_{Path(source_name).stem}.jpg",
            mime="image/jpeg",
            use_container_width=False,
        )


# -----------------------------------------------------------------------------
# Module C: PPE Video Detection
# -----------------------------------------------------------------------------
def render_video_detection_module(config: Dict[str, Any], resolved_model_path: str, device_option: str, conf_threshold: float, frame_stride: int):
    """
    Renders Section B of Phase 5: Video processing, frame sampling, temporal tracking,
    annotated playback, compliance timeline chart, and CSV/JSON downloads.
    """
    st.markdown('<div class="main-header">🎬 PPE Video Detection & Compliance Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Frame Temporal Tracking and PPE Compliance Video Analytics for Active Construction Sites</div>',
        unsafe_allow_html=True,
    )

    st.subheader("1. Video Upload")
    uploaded_file = st.file_uploader(
        "Upload a construction site video file",
        type=["mp4", "avi", "mov"],
        help="Upload standard MP4, AVI, or MOV construction site footage.",
        key="ppe_video_uploader",
    )

    if uploaded_file is None:
        st.info("👋 Upload a video file above to begin PPE compliance analysis. (Max size: 500 MB)")
        return

    # Save uploaded file to secure temporary workspace
    temp_dir = Path(tempfile.gettempdir()) / "safeedge_ppe_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_video_path = temp_dir / f"upload_{uploaded_file.name}"

    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 2: Validate Video Metadata
    max_mb = float(config.get("video", {}).get("max_upload_size_mb", 500.0))
    metadata = extract_video_metadata(temp_video_path, max_size_mb=max_mb)

    if not metadata.is_valid:
        st.error(f"❌ Video Validation Failed: {metadata.error_message}")
        return

    st.success(f"✅ Video '{metadata.file_name}' validated successfully.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Resolution", f"{metadata.width} × {metadata.height}")
    with col2:
        st.metric("Duration", metadata.duration_str)
    with col3:
        st.metric("Frame Rate", f"{metadata.fps:.1f} FPS")
    with col4:
        st.metric("Total Frames", f"{metadata.total_frames:,}")
    with col5:
        st.metric("File Size", f"{metadata.file_size_mb:.1f} MB")

    # Step 3: Processing Action
    st.divider()
    st.subheader("2. Run PPE Compliance Analysis")

    cols_action1, cols_action2 = st.columns([1, 4])
    with cols_action1:
        process_clicked = st.button("🚀 Process Video", type="primary", use_container_width=True, key="btn_run_video_proc")

    with cols_action2:
        frames_to_process = (metadata.total_frames + frame_stride - 1) // frame_stride
        st.caption(
            f"Configured to process **{frames_to_process}** sampled frames "
            f"(Stride: {frame_stride}, Conf: {conf_threshold:.2f}, Device: {device_option})."
        )

    session_key = f"results_{uploaded_file.name}_{metadata.total_frames}_{frame_stride}_{conf_threshold}"

    if process_clicked:
        try:
            detector, _ = get_cached_detector(resolved_model_path, device_option)
        except Exception as e:
            st.error(f"❌ Failed to load PPE detection model: {e}")
            return

        assessor = PPEAssessor(config=config)
        visualizer = PPEVisualizer(config=config)
        tracker = SimpleTracker()

        processor = VideoProcessor(
            detector=detector,
            assessor=assessor,
            visualizer=visualizer,
            tracker=tracker,
            config=config,
        )

        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_output_video = output_dir / f"annotated_{Path(uploaded_file.name).stem}.mp4"

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(current, total, pct, msg):
            progress_bar.progress(pct)
            status_text.text(f"⏳ {msg} ({pct * 100:.1f}%)")

        try:
            with st.spinner("Processing video frames and evaluating PPE compliance..."):
                final_video_path, stats = processor.process_video(
                    video_path=temp_video_path,
                    output_path=target_output_video,
                    frame_stride=frame_stride,
                    conf_threshold=conf_threshold,
                    progress_callback=on_progress,
                )

            progress_bar.progress(1.0)
            status_text.success("🎉 Video processing complete!")

            # Store in session state
            res_dict = {
                "final_video_path": final_video_path,
                "stats": stats,
                "metadata": metadata,
            }
            st.session_state[session_key] = res_dict
            st.session_state["last_video_results"] = res_dict

        except Exception as err:
            st.error(f"❌ An error occurred during video processing: {err}")
            return

    # Step 4: Display Results & Analytics
    if session_key in st.session_state:
        res = st.session_state[session_key]
        stats: VideoProcessingStats = res["stats"]
        meta: VideoMetadata = res["metadata"]
        video_path = res["final_video_path"]

        st.divider()
        st.subheader("3. Video Playback & Compliance Analytics")

        v_col, s_col = st.columns([3, 2])

        with v_col:
            st.markdown("##### 🎬 Annotated Output Video")
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                st.video(video_path)
            else:
                st.warning("Video file not found or empty.")

        with s_col:
            st.markdown("##### 📊 Processing Performance")
            st.markdown(
                f"""
                - **Frames Analyzed:** `{stats.frames_processed}` of `{stats.total_frames_in_video}` frames (Stride: `{stats.sampling_stride}`)
                - **Processing Time:** `{stats.processing_time_sec:.2f} s`
                - **Throughput:** `{stats.fps_throughput:.1f} FPS`
                - **Total Person Observations:** `{stats.total_person_observations}` (frame-level sightings)
                - **Unique Tracked Workers:** `{stats.unique_tracked_workers}` (distinct individuals)
                """
            )

            total_obs = max(1, stats.total_person_observations)
            comp_rate = (stats.compliant_observations / total_obs) * 100
            viol_rate = (stats.violation_observations / total_obs) * 100
            rev_rate = (stats.review_required_observations / total_obs) * 100

            st.markdown("##### 🛡️ Worker Compliance Breakdown")
            st.progress(comp_rate / 100.0, text=f"Compliant Observations: {comp_rate:.1f}%")
            st.progress(viol_rate / 100.0, text=f"Violation Observations: {viol_rate:.1f}%")
            st.progress(rev_rate / 100.0, text=f"Review Required: {rev_rate:.1f}%")

        st.markdown("##### 📈 Detailed PPE Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)

        with c1:
            st.metric("Frames Processed", f"{stats.frames_processed:,}")
        with c2:
            st.metric("Person Observations", f"{stats.total_person_observations:,}")
        with c3:
            st.metric("Unique Tracked Workers", f"{stats.unique_tracked_workers}")
        with c4:
            st.metric("Compliant Observations", f"{stats.compliant_observations:,}")

        with c5:
            st.metric("Helmet Positive Obs", f"{stats.helmet_positive_observations:,}")
        with c6:
            st.metric("Vest Positive Obs", f"{stats.vest_positive_observations:,}")
        with c7:
            st.metric("Helmet Violations", f"{stats.helmet_violation_observations:,}")
        with c8:
            st.metric("Vest Violations", f"{stats.vest_violation_observations:,}")

        st.divider()
        st.subheader("4. Safety Compliance Timeline")

        if stats.timeline_data:
            df_timeline = pd.DataFrame(stats.timeline_data)
            df_chart = df_timeline.set_index("timestamp_str")[
                ["compliant_count", "violation_count", "review_count"]
            ]
            df_chart.columns = ["Compliant Workers", "Violation Events", "Review Required"]

            st.line_chart(df_chart, use_container_width=True, color=["#22C55E", "#EF4444", "#F97316"])
            st.caption("Timeline shows worker observations per sampled frame over time (seconds).")

        st.divider()
        st.subheader("5. Worker Observation & Violation Events Log")

        if stats.events:
            df_events = pd.DataFrame(stats.events)

            filter_col1, filter_col2 = st.columns([2, 3])
            with filter_col1:
                status_filter = st.selectbox(
                    "Filter by Compliance Status",
                    options=["All Observations", "VIOLATION Only", "REVIEW_REQUIRED Only", "COMPLIANT Only"],
                    key="select_filter_video_status",
                )

            filtered_df = df_events
            if status_filter == "VIOLATION Only":
                filtered_df = df_events[df_events["overall_status"] == "VIOLATION"]
            elif status_filter == "REVIEW_REQUIRED Only":
                filtered_df = df_events[df_events["overall_status"] == "REVIEW_REQUIRED"]
            elif status_filter == "COMPLIANT Only":
                filtered_df = df_events[df_events["overall_status"] == "COMPLIANT"]

            display_cols = [
                "timestamp_str", "worker_id", "overall_status", "helmet_status",
                "vest_status", "person_confidence", "explanation"
            ]
            st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(filtered_df)} of {len(df_events)} total recorded observations.")
        else:
            st.info("No worker observations recorded.")

        st.divider()
        st.subheader("6. Download Reports & Annotated Media")

        csv_path = PROJECT_ROOT / "output" / f"report_{Path(uploaded_file.name).stem}.csv"
        json_path = PROJECT_ROOT / "output" / f"summary_{Path(uploaded_file.name).stem}.json"

        export_csv_report(stats, csv_path)
        export_json_report(
            metadata=meta,
            stats=stats,
            json_path=json_path,
            model_info={
                "model_identifier": "Hansung-Cho/yolov8-ppe-detection",
                "weights_file": resolved_model_path,
                "base_architecture": "Ultralytics YOLOv8n",
                "license": "MIT",
                "framework": "PyTorch + Ultralytics",
            },
            settings={
                "confidence_threshold": conf_threshold,
                "sampling_stride": frame_stride,
                "device": device_option,
            },
        )

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                with open(video_path, "rb") as vf:
                    st.download_button(
                        label="📥 Download Annotated Video (MP4)",
                        data=vf.read(),
                        file_name=f"annotated_{Path(uploaded_file.name).stem}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

        with dl_col2:
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as cf:
                    st.download_button(
                        label="📥 Download Audit Report (CSV)",
                        data=cf.read(),
                        file_name=f"ppe_report_{Path(uploaded_file.name).stem}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with dl_col3:
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as jf:
                    st.download_button(
                        label="📥 Download Summary Report (JSON)",
                        data=jf.read(),
                        file_name=f"ppe_summary_{Path(uploaded_file.name).stem}.json",
                        mime="application/json",
                        use_container_width=True,
                    )


# -----------------------------------------------------------------------------
# Main Application Entry Point & Navigation
# -----------------------------------------------------------------------------
def main():
    inject_custom_styles()

    # Load configuration
    config = ModelLoader.load_config()
    default_model_path = config.get("model", {}).get("model_path", "models/best.pt")
    resolved_model_path = str((PROJECT_ROOT / default_model_path).resolve())

    # Initialize or retrieve persistent environmental simulator
    sim = get_or_create_simulator()

    # -------------------------------------------------------------------------
    # Unified Sidebar Navigation & Global Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🛡️ SafeEdge AI Navigation")

        # Map legacy selection values if set previously in session
        nav_options = [
            "🏢 Executive Safety Overview",
            "📷 PPE Image Detection",
            "🎬 PPE Video Detection",
            "🏗️ Environmental Hazard Simulator",
            "📊 Hazard History & Reports",
        ]

        current_nav = st.session_state.get("safeedge_main_module_nav", "🏢 Executive Safety Overview")
        if current_nav not in nav_options:
            if "Environmental" in current_nav:
                current_nav = "🏗️ Environmental Hazard Simulator"
            else:
                current_nav = "🏢 Executive Safety Overview"

        selected_module = st.radio(
            "System Module Selection",
            nav_options,
            index=nav_options.index(current_nav),
            key="safeedge_main_module_nav",
            help="Navigate between Executive Overview, PPE Detection, Environmental Simulation, and Reports."
        )

        st.divider()

        # Shared Vision Settings (shown on vision inspection modules)
        if "PPE" in selected_module or "Overview" in selected_module:
            st.header("⚙️ Vision Model Settings")

            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.15,
                max_value=0.85,
                value=float(config.get("model", {}).get("confidence_threshold", 0.35)),
                step=0.05,
                help="Minimum probability score required to accept detections.",
                key="global_conf_threshold",
            )

            frame_stride = st.select_slider(
                "Video Sampling Stride",
                options=[1, 2, 3, 5, 10, 15],
                value=int(config.get("video", {}).get("default_stride", 2)),
                help="Frame sampling frequency for video inspection.",
                key="global_frame_stride",
            )

            device_option = st.selectbox(
                "Compute Device",
                options=["auto", "cpu", "cuda"],
                index=0,
                help="Auto selects CUDA GPU when available, otherwise CPU.",
                key="global_device_option",
            )

            st.divider()
            st.subheader("🤖 Verified Model Info")
            st.markdown(
                """
                - **Model:** `Hansung-Cho/yolov8-ppe-detection`
                - **Architecture:** YOLOv8n (Nano)
                - **Weights:** `models/best.pt` (~6.25 MB)
                - **Classes:** `Person`, `Hardhat`, `Safety Vest`, `NO-Hardhat`, `NO-Safety Vest`
                """
            )
        else:
            conf_threshold = float(config.get("model", {}).get("confidence_threshold", 0.35))
            frame_stride = int(config.get("video", {}).get("default_stride", 2))
            device_option = "auto"

        st.caption("SafeEdge AI v2.0 • Academic Demonstration Edition")

    # -------------------------------------------------------------------------
    # Route to Selected Module
    # -------------------------------------------------------------------------
    if selected_module == "🏢 Executive Safety Overview":
        render_overview_module(sim)

    elif selected_module == "📷 PPE Image Detection":
        render_image_detection_module(config, resolved_model_path, device_option, conf_threshold)

    elif selected_module == "🎬 PPE Video Detection":
        render_video_detection_module(config, resolved_model_path, device_option, conf_threshold, frame_stride)

    elif selected_module == "🏗️ Environmental Hazard Simulator":
        render_simulator_view(sim)

    elif selected_module == "📊 Hazard History & Reports":
        render_history_reports_view(sim)


if __name__ == "__main__":
    main()
