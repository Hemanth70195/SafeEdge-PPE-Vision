"""
SafeEdge PPE Vision - Professional Environmental Hazard Monitor Dashboard.
Phase 3: Professional construction-site safety dashboard designed for high-clarity demonstration.

Sections:
A. Overall Site Status: Large banner with status (GREEN/ORANGE/RED), active warning/danger counts,
   and root-cause diagnostic explanation.
B. Five Environmental Hazard Cards: Real-time cards displaying reading, unit, status, icons,
   configured thresholds, diagnostic reasons, and Phase 2 protective action recommendations.
C. Simulation Controls: Manual vs Automatic mode, Start, Pause, Resume, Reset, Single-Step, Interval control.
D. Manual Controls: Intuitive parameter sliders with configured bounds and demonstration scenario presets.
E. Hazard History: Time-series charts with un-mixed units, multi-tab visualization, and empty-state handling.
F. Hazard Event Log: Deduplicated warning and danger event feed with timestamps, values, severity, and actions.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd

from core.environmental_simulator import (
    EnvironmentalSimulator,
    EnvironmentalReading,
    HazardStatus,
    SIMULATED_DATA_DISCLAIMER_SHORT,
    DEFAULT_CONFIG_PATH,
)
from core.hazard_assessment import (
    HazardAssessmentEngine,
    HazardAssessment,
    SiteHazardAssessment,
)
from core.hazard_history import (
    HazardEvent,
    HazardHistoryTracker,
)

# Prominent Phase 3 Required Disclaimers
PHASE3_DISCLAIMER_TITLE = "SIMULATION MODE â NOT LIVE SENSOR DATA"
PHASE3_DISCLAIMER_SUB = (
    "All environmental readings, hazard states, and recommendations are simulated in software "
    "for training, testing, and design demonstration. Simulated readings do NOT certify actual site safety."
)


def get_or_create_simulator() -> EnvironmentalSimulator:
    """Initializes or retrieves the persistent simulator instance in Streamlit session state."""
    if "env_simulator" not in st.session_state:
        st.session_state.env_simulator = EnvironmentalSimulator(config_path=DEFAULT_CONFIG_PATH)

        # Initialize session state for manual slider values
        defaults = st.session_state.env_simulator.get_default_readings()
        for k, v in defaults.items():
            st.session_state[f"env_slider_{k}"] = float(v)

        st.session_state.env_sim_mode = "manual"
        st.session_state.env_is_auto_running = False
        st.session_state.env_is_paused = False
        st.session_state.env_auto_interval = 1.5

    return st.session_state.env_simulator


def render_dashboard_styles():
    """Injects custom CSS for a professional, modern construction-safety aesthetic."""
    st.markdown(
        """
        <style>
        /* Base typography & container sizing */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .monitor-header {
            background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
            border-radius: 12px;
            padding: 20px 24px;
            color: #FFFFFF;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #38BDF8;
        }
        .monitor-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #F8FAFC;
        }
        .monitor-subtitle {
            font-size: 0.96rem;
            color: #94A3B8;
            margin-top: 6px;
            margin-bottom: 0;
            line-height: 1.4;
        }
        .sim-disclaimer-banner {
            background-color: #FEF3C7;
            border-left: 5px solid #F59E0B;
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 700;
            color: #92400E;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        /* Section A: Overall site status banner */
        .overall-status-card {
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 1.5rem;
            border: 2px solid;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.25s ease;
        }
        .overall-status-safe {
            background-color: #ECFDF5;
            border-color: #10B981;
            color: #065F46;
        }
        .overall-status-warning {
            background-color: #FFFBEB;
            border-color: #F59E0B;
            color: #92400E;
        }
        .overall-status-danger {
            background-color: #FEF2F2;
            border-color: #EF4444;
            color: #991B1B;
        }
        .overall-status-header {
            font-size: 1.4rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
            letter-spacing: -0.01em;
        }
        .overall-counts-row {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            font-size: 0.95rem;
            font-weight: 700;
        }
        .count-pill-danger {
            background-color: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FECACA;
            padding: 4px 12px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .count-pill-warning {
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #FDE68A;
            padding: 4px 12px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .count-pill-safe {
            background-color: #D1FAE5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 4px 12px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .overall-summary-text {
            font-size: 0.98rem;
            margin: 0;
            line-height: 1.45;
            font-weight: 500;
        }
        /* Section B: Parameter Cards */
        .param-card-v3 {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            border: 2px solid #E2E8F0;
            margin-bottom: 0.8rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .param-card-header {
            font-size: 0.82rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .param-card-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
            margin: 6px 0;
        }
        .param-card-unit {
            font-size: 0.95rem;
            font-weight: 600;
            color: #64748B;
            margin-left: 4px;
        }
        .threshold-pill {
            font-size: 0.72rem;
            color: #475569;
            background-color: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 3px 6px;
            margin: 6px 0;
            line-height: 1.3;
        }
        .card-explanation-box {
            font-size: 0.82rem;
            color: #334155;
            margin-top: 6px;
            line-height: 1.35;
            min-height: 2.8em;
        }
        .card-action-box {
            font-size: 0.80rem;
            color: #1E293B;
            background-color: #F8FAFC;
            border-left: 3px solid #64748B;
            padding: 6px 10px;
            border-radius: 0 6px 6px 0;
            margin-top: 8px;
            line-height: 1.35;
        }
        .card-footer-disclaimer {
            font-size: 0.65rem;
            color: #94A3B8;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
            border-top: 1px dashed #E2E8F0;
            padding-top: 6px;
        }
        /* Active Hazard Banner Box */
        .active-hazard-card {
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
            border-left: 6px solid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .active-hazard-danger {
            border-left-color: #EF4444;
            background-color: #FEF2F2;
        }
        .active-hazard-warning {
            border-left-color: #F59E0B;
            background-color: #FFFBEB;
        }
        .active-hazard-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0F172A;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        /* Badges */
        .badge-green {
            background-color: #D1FAE5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.78rem;
            display: inline-block;
        }
        .badge-orange {
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #FDE68A;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.78rem;
            display: inline-block;
        }
        .badge-red {
            background-color: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FECACA;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.78rem;
            display: inline-block;
        }
        .section-header-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 8px;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        .section-header-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0F172A;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-card-v3 {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 18px;
            margin-top: 14px;
            margin-bottom: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_threshold_summary(meta: Dict[str, Any]) -> str:
    """Produces a concise summary string of the configured warning and danger thresholds."""
    t = meta.get("thresholds", {})
    unit = meta.get("unit", "")
    ttype = t.get("type", "")

    if ttype == "range":
        s_min = t.get("safe_min", "")
        s_max = t.get("safe_max", "")
        w_min = t.get("warning_min", "")
        w_max = t.get("warning_max", "")
        return f"Safe: {s_min}â{s_max} {unit} | Warn: <{s_min} or >{s_max} | Danger: <{w_min} or >{w_max}"
    elif ttype == "upper":
        w = t.get("warning_threshold", "")
        d = t.get("danger_threshold", "")
        return f"Safe: â¤ {w} {unit} | Warning: > {w} {unit} | Danger: > {d} {unit}"
    return "Configured dynamically"


def render_overall_site_status(site_ass: SiteHazardAssessment):
    """
    Renders Section A: Large prominent status panel showing:
    - Overall status (GREEN, ORANGE, RED with clear text labels).
    - Number of active warnings.
    - Number of critical hazards.
    - Short explanation of the most severe active hazard.
    """
    if site_ass.overall_status == HazardStatus.SAFE:
        css_class = "overall-status-safe"
        icon = "✅"
        status_label = "SITE STATUS: GREEN â NORMAL & SAFE"
    elif site_ass.overall_status == HazardStatus.WARNING:
        css_class = "overall-status-warning"
        icon = "⚠️"
        status_label = "SITE STATUS: ORANGE â ELEVATED WARNING"
    else:
        css_class = "overall-status-danger"
        icon = "🚨"
        status_label = "SITE STATUS: RED â CRITICAL HAZARD DANGER"

    st.markdown(
        f"""
        <div class="overall-status-card {css_class}">
            <div class="overall-status-header">
                <span>{icon}</span>
                <span>{status_label}</span>
            </div>
            <div class="overall-counts-row">
                <span class="count-pill-danger">🔴 Critical Hazards: {site_ass.danger_count}</span>
                <span class="count-pill-warning">🟠 Active Warnings: {site_ass.warning_count}</span>
                <span class="count-pill-safe">🟢 Safe Parameters: {len(site_ass.safe_parameters)} / 5</span>
            </div>
            <p class="overall-summary-text"><strong>Site Safety Summary:</strong> {site_ass.overall_summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_active_hazard_prioritized_list(site_ass: SiteHazardAssessment):
    """
    Renders the active hazards ordered strictly by severity (DANGER first, then WARNING).
    Displays the explanation and Phase 2 recommended protective actions.
    """
    if not site_ass.has_hazards:
        st.success(
            "✅ **All 5 Environmental Parameters Within Safe Limits:** "
            "No active safety interventions required. Maintain routine site vigilance."
        )
        return

    icons = {
        "temperature": "🌡️",
        "humidity": "💧",
        "dust": "🏗️",
        "noise": "🔊",
        "gas": "☣️",
    }

    st.markdown("##### 🛡️ Active Hazard Severity Prioritization & Protective Directives")

    for h in site_ass.active_hazards:
        icon = icons.get(h.parameter_key, "📊")
        box_class = "active-hazard-danger" if h.status == HazardStatus.DANGER else "active-hazard-warning"
        badge_html = (
            '<span class="badge-red">🔴 RED â DANGER</span>'
            if h.status == HazardStatus.DANGER
            else '<span class="badge-orange">🟠 ORANGE â WARNING</span>'
        )

        actions_bullets = "".join(f"<li>{act}</li>" for act in h.recommended_actions)

        st.markdown(
            f"""
            <div class="active-hazard-card {box_class}">
                <div class="active-hazard-title">
                    <span>{icon} {h.display_name}: <strong>{h.value:.1f} {h.unit}</strong></span>
                    {badge_html}
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-bottom: 8px;">
                    <strong>Diagnostic Reason:</strong> {h.reason}
                </div>
                <div style="font-size: 0.80rem; font-weight: 800; text-transform: uppercase; color: #475569; margin-bottom: 4px;">
                    Recommended Protective Actions:
                </div>
                <ul style="margin: 0; padding-left: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.45;">
                    {actions_bullets}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hazard_card_v3(assessment: HazardAssessment, meta: Dict[str, Any]):
    """
    Renders Section B: Individual hazard card displaying:
    - Current reading and unit.
    - GREEN, ORANGE, or RED status (text label + badge + icon).
    - Configured warning and danger thresholds.
    - Explanation of why status was assigned.
    - Recommended protective action supplied by Phase 2.
    """
    if assessment.status == HazardStatus.SAFE:
        badge_html = '<span class="badge-green">🟢 GREEN â SAFE</span>'
        border_color = "#10B981"
    elif assessment.status == HazardStatus.WARNING:
        badge_html = '<span class="badge-orange">🟠 ORANGE â WARNING</span>'
        border_color = "#F59E0B"
    else:
        badge_html = '<span class="badge-red">🔴 RED â DANGER</span>'
        border_color = "#EF4444"

    icons = {
        "temperature": "🌡️",
        "humidity": "💧",
        "dust": "🏗️",
        "noise": "🔊",
        "gas": "☣️",
    }
    icon = icons.get(assessment.parameter_key, "📊")
    threshold_str = format_threshold_summary(meta)
    primary_action = assessment.recommended_actions[0] if assessment.recommended_actions else "Standard monitoring."

    st.markdown(
        f"""
        <div class="param-card-v3" style="border-color: {border_color};">
            <div>
                <div class="param-card-header">{icon} {assessment.display_name}</div>
                <div class="param-card-value">{assessment.value:.1f}<span class="param-card-unit">{assessment.unit}</span></div>
                <div style="margin-bottom: 6px;">{badge_html}</div>
                <div class="threshold-pill"><strong>Thresholds:</strong> {threshold_str}</div>
                <div class="card-explanation-box"><strong>Diagnosis:</strong> {assessment.reason}</div>
                <div class="card-action-box"><strong>Key Action:</strong> {primary_action}</div>
            </div>
            <div class="card-footer-disclaimer">⚠️ {SIMULATED_DATA_DISCLAIMER_SHORT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_section(sim: EnvironmentalSimulator):
    """
    Renders Section E: Hazard history showing recent simulated readings during the session.
    Features:
    - Status filtering (All, Safe, Warning, Danger) and record count limit.
    - Separate time-series charts for distinct units to avoid misleading multi-unit overlay.
    - Formatted readings table with individual and overall statuses.
    - Useful empty-state messaging.
    """
    st.markdown(
        """
        <div class="section-header-box">
            <h3 class="section-header-title">📈 Section E: Hazard History & Telemetry Analysis</h3>
            <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Session In-Memory Buffer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_snaps = sim.get_history_snapshots_df()

    if df_snaps.empty or len(df_snaps) <= 1:
        st.info(
            "📊 **No Historical Telemetry Recorded Yet:** "
            "Session history begins recording as simulation steps progress. "
            "Adjust any manual slider or click **Start Automatic Simulation** to generate real-time time-series trends."
        )
        return

    # Sub-tabs for History: Charts vs Filtered Readings Table
    hist_view_tab1, hist_view_tab2 = st.tabs([
        "📈 Time-Series Trend Charts",
        "📋 Historical Readings Data Table",
    ])

    with hist_view_tab1:
        st.caption("Categorized multi-axis time-series trends across distinct engineering unit domains:")
        chart_tab1, chart_tab2, chart_tab3 = st.tabs([
            "🌡️ Microclimate (°C & %)",
            "🏗️ Particulate & Acoustic (Dust & Noise)",
            "☣️ Gas Index (ppm)",
        ])

        with chart_tab1:
            st.caption("Ambient Dry-Bulb Temperature (°C) and Relative Atmospheric Humidity (%):")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("**Temperature (°C)**")
                st.line_chart(df_snaps.set_index("Timestamp")["temperature"], color="#EF4444", use_container_width=True)
            with sub_col2:
                st.markdown("**Relative Humidity (%)**")
                st.line_chart(df_snaps.set_index("Timestamp")["humidity"], color="#3B82F6", use_container_width=True)

        with chart_tab2:
            st.caption("Respirable Dust Concentration (µg/m³) and Acoustic Noise Level (dB):")
            sub_col3, sub_col4 = st.columns(2)
            with sub_col3:
                st.markdown("**Dust Concentration (µg/m³ PM10)**")
                st.line_chart(df_snaps.set_index("Timestamp")["dust"], color="#F59E0B", use_container_width=True)
            with sub_col4:
                st.markdown("**Noise Level (dB)**")
                st.line_chart(df_snaps.set_index("Timestamp")["noise"], color="#8B5CF6", use_container_width=True)

        with chart_tab3:
            st.caption("Hazardous Airborne Toxic / Combustible Gas Concentration Index (ppm):")
            st.line_chart(df_snaps.set_index("Timestamp")["gas"], color="#DC2626", use_container_width=True)

    with hist_view_tab2:
        st.caption("Filter and inspect recorded telemetry snapshots:")
        f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2])
        with f_col1:
            status_filter = st.selectbox(
                "Filter by Overall Status",
                options=["All Records", "🟢 Safe Only", "🟠 Warning Active", "🔴 Danger Active"],
                key="filter_history_status",
            )
        with f_col2:
            row_limit = st.selectbox(
                "Display Count",
                options=["Last 20", "Last 50", "Last 100", "All Records"],
                index=1,
                key="filter_history_rows",
            )
        with f_col3:
            st.markdown(f"**Total In-Memory Buffer:** `{len(df_snaps)}` snapshots recorded during session.")

        # Apply filtering
        filtered_df = df_snaps.copy()
        if status_filter == "🟢 Safe Only":
            filtered_df = filtered_df[filtered_df["overall_status"] == "SAFE"]
        elif status_filter == "🟠 Warning Active":
            filtered_df = filtered_df[filtered_df["overall_status"] == "WARNING"]
        elif status_filter == "🔴 Danger Active":
            filtered_df = filtered_df[filtered_df["overall_status"] == "DANGER"]

        if row_limit == "Last 20":
            filtered_df = filtered_df.tail(20)
        elif row_limit == "Last 50":
            filtered_df = filtered_df.tail(50)
        elif row_limit == "Last 100":
            filtered_df = filtered_df.tail(100)

        # Reverse to show newest at top for user inspection
        display_df = filtered_df.iloc[::-1].copy()

        # Display readable columns
        readable_cols = {
            "Full_Timestamp": "Recorded At",
            "Tick": "Tick",
            "Mode": "Mode",
            "temperature": "Temp (°C)",
            "humidity": "Humidity (%)",
            "dust": "Dust (µg/m³)",
            "noise": "Noise (dB)",
            "gas": "Gas (ppm)",
            "overall_status": "Overall Status",
        }
        present_cols = [c for c in readable_cols.keys() if c in display_df.columns]
        table_to_show = display_df[present_cols].rename(columns={c: readable_cols[c] for c in present_cols})

        if table_to_show.empty:
            st.warning("No records match the selected status filter.")
        else:
            st.dataframe(table_to_show, use_container_width=True, hide_index=True)


def render_event_log_section(sim: EnvironmentalSimulator):
    """
    Renders Section F: Hazard event timeline and transition log showing:
    - Timestamp.
    - Hazard type.
    - State transition (GREEN -> ORANGE, ORANGE -> RED, RED -> GREEN, etc.).
    - Observed value and unit.
    - Relevant threshold crossed.
    - Diagnostic reason.
    - Recommended protective action.
    - Interactive filtering by hazard type and severity.
    """
    st.markdown(
        """
        <div class="section-header-box">
            <h3 class="section-header-title">📋 Section F: Hazard State-Transition Timeline & Event Log</h3>
            <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Deduplicated Threshold Transitions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_events = sim.get_hazard_events_df()

    if df_events.empty:
        st.success(
            "✅ **No Hazard State Transitions Recorded:** "
            "Site is operating within normal baseline safety limits. "
            "State transitions (Warning, Danger, Recovery) will be logged here automatically."
        )
        return

    # Filter controls for events
    ef_col1, ef_col2, ef_col3 = st.columns([1.5, 1.5, 2])
    with ef_col1:
        param_filter = st.selectbox(
            "Filter by Parameter",
            options=["All Parameters", "Temperature", "Relative Humidity", "Dust Concentration", "Noise Level", "Gas Concentration"],
            key="filter_event_param",
        )
    with ef_col2:
        sev_filter = st.selectbox(
            "Filter by Severity / Transition",
            options=["All Events", "🔴 Critical / Danger", "🟠 Warning Active", "🟢 Recovery to Safe"],
            key="filter_event_sev",
        )
    with ef_col3:
        st.markdown(f"**Total Events in Session:** `{len(df_events)}` transitions documented.")

    # Apply filters
    filtered_events = df_events.copy()
    if param_filter != "All Parameters":
        filtered_events = filtered_events[filtered_events["Hazard Type"].str.contains(param_filter, case=False, na=False)]

    if sev_filter == "🔴 Critical / Danger":
        filtered_events = filtered_events[filtered_events["Severity"].str.contains("RED|DANGER", case=False, na=False)]
    elif sev_filter == "🟠 Warning Active":
        filtered_events = filtered_events[filtered_events["Severity"].str.contains("ORANGE|WARNING", case=False, na=False)]
    elif sev_filter == "🟢 Recovery to Safe":
        filtered_events = filtered_events[filtered_events["Severity"].str.contains("GREEN|SAFE", case=False, na=False)]

    if filtered_events.empty:
        st.info("No hazard events match the chosen parameter and severity criteria.")
    else:
        st.caption(f"Displaying **{len(filtered_events)}** documented events (newest first):")
        st.dataframe(filtered_events, use_container_width=True, hide_index=True)


def render_summary_statistics_section(sim: EnvironmentalSimulator):
    """
    Renders Section G: Session Summary Statistics & Exposure Metrics:
    - Min, Max, Average reading for each of the five parameters.
    - Warning, critical, and recovery event counts.
    - Session duration and time range.
    - Prominent simulation disclaimer.
    """
    st.markdown(
        """
        <div class="section-header-box">
            <h3 class="section-header-title">📊 Section G: Telemetry Summary Statistics</h3>
            <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Active Session Aggregations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = sim.get_summary_statistics()

    if not stats.get("has_data", False):
        st.info("📊 **No Telemetry Recorded Yet:** Statistical metrics will populate as simulation snapshots are captured.")
        return

    # Simulation Notice
    st.caption(f"⚠️ **Simulation Statistics Notice:** {stats.get('disclaimer', '')}")

    # Session Metrics Banner
    t_range = stats.get("time_range", {})
    ev_summary = stats.get("events_summary", {})

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Session Duration", t_range.get("duration_formatted", "00:00:00"), help="Active elapsed simulation session duration.")
    with m_col2:
        st.metric("Total Snapshots", stats.get("total_snapshots", 0), help="Number of telemetry snapshots recorded in session buffer.")
    with m_col3:
        st.metric("🔴 Critical Events", ev_summary.get("critical_events", 0), delta=f"{ev_summary.get('critical_events', 0)} DANGER", delta_color="inverse")
    with m_col4:
        st.metric("🟠 Warning Events", ev_summary.get("warning_events", 0), delta=f"{ev_summary.get('warning_events', 0)} WARNING", delta_color="inverse")
    with m_col5:
        st.metric("🟢 Recoveries", ev_summary.get("recovery_events", 0), delta=f"{ev_summary.get('recovery_events', 0)} SAFE", delta_color="normal")

    st.markdown("##### Parameter Aggregate Statistics (Min / Max / Mean)")

    # Build parameter summary table
    param_data = stats.get("parameters", {})
    rows = []
    for k, p in param_data.items():
        rows.append({
            "Environmental Parameter": p.get("name", k),
            "Unit": p.get("unit", ""),
            "Minimum Reading": f"{p.get('min', 0.0):.2f}",
            "Maximum Reading": f"{p.get('max', 0.0):.2f}",
            "Mean / Average Reading": f"{p.get('mean', 0.0):.2f}",
            "Valid Samples": p.get("count", 0),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_report_export_section(sim: EnvironmentalSimulator):
    """
    Renders Section H: Compliance & Audit Report Export:
    - CSV download for historical readings.
    - CSV download for hazard state-transition events.
    - JSON audit report download.
    - Preview expander.
    """
    import datetime

    st.markdown(
        """
        <div class="section-header-box">
            <h3 class="section-header-title">💾 Section H: Compliance & Audit Report Export</h3>
            <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Multi-Format Report Generation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Download documented simulation session logs for project audit, analysis, and academic demonstration.
        All exported files include standardized disclaimers and metadata.
        """
    )

    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate payloads
    readings_csv_data = sim.export_readings_csv()
    events_csv_data = sim.export_events_csv()
    json_report_data = sim.export_json_report_str()

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        st.download_button(
            label="📥 Download Readings CSV",
            data=readings_csv_data,
            file_name=f"safeedge_sim_readings_{ts_str}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download time-stamped environmental parameter readings, individual statuses, and overall status in CSV format.",
        )
        st.caption("Includes: Timestamp, 5 Parameters with Units, Statuses, Simulation Mode.")

    with btn_col2:
        st.download_button(
            label="📥 Download Hazard Events CSV",
            data=events_csv_data,
            file_name=f"safeedge_sim_events_{ts_str}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download all hazard state transitions, threshold crossings, diagnostic reasons, and protective actions.",
        )
        st.caption("Includes: Timestamp, Transitions, Observed Values, Thresholds, Actions.")

    with btn_col3:
        st.download_button(
            label="📥 Download JSON Audit Report",
            data=json_report_data,
            file_name=f"safeedge_sim_audit_report_{ts_str}.json",
            mime="application/json",
            use_container_width=True,
            help="Download complete structured JSON audit report containing metadata, threshold configurations, statistics, and full logs.",
        )
        st.caption("Includes: Full schema, threshold settings, statistics, readings, and events.")

    with st.expander("🔍 Preview Structured JSON Report Payload"):
        st.code(json_report_data[:1200] + "\n... [truncated for preview] ...", language="json")


def render_legend():
    """Renders the status explanation legend card."""
    st.markdown(
        """
        <div class="legend-card-v3">
            <div style="font-weight: 800; font-size: 0.90rem; color: #0F172A; margin-bottom: 6px;">
                📖 Status Classification Legend
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.86rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="badge-green">🟢 GREEN â SAFE</span>
                    <span style="color: #475569;">Within configured acceptable range. Normal vigilance applies.</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="badge-orange">🟠 ORANGE â WARNING</span>
                    <span style="color: #475569;">Threshold exceeded; precautionary safety interventions recommended.</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="badge-red">🔴 RED â DANGER</span>
                    <span style="color: #475569;">Critical hazard threshold exceeded; urgent emergency action required.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simulator_view(sim: Optional[EnvironmentalSimulator] = None):
    """
    Renders the interactive Environmental Hazard Simulator cockpit:
    - Overall site status banner (Section A)
    - Active hazard severity directives
    - Simulation mode controls & manual parameter sliders with presets (Sections C & D)
    - Five real-time environmental parameter cards (Section B)
    - Status legend and threshold reference matrix
    - Controlled auto-simulation loop
    """
    render_dashboard_styles()

    if sim is None:
        sim = get_or_create_simulator()

    # Professional Header & Prominent Disclaimers
    st.markdown(
        f"""
        <div class="monitor-header">
            <h1 class="monitor-title">🏗️ SafeEdge Environmental Hazard Simulator</h1>
            <p class="monitor-subtitle">
                Real-Time Construction Site Microclimate, Particulate Dust, Noise Exposure, and Hazardous Gas Monitoring Cockpit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sim-disclaimer-banner">
            <span>⚠️</span>
            <span><strong>{PHASE3_DISCLAIMER_TITLE}:</strong> {PHASE3_DISCLAIMER_SUB}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Evaluation of Current State
    # -------------------------------------------------------------------------
    site_assessment = sim.assess_current_hazards()

    # -------------------------------------------------------------------------
    # Section A: Overall Site Status
    # -------------------------------------------------------------------------
    render_overall_site_status(site_assessment)

    # -------------------------------------------------------------------------
    # Section C & D: Simulation Controls & Manual Controls
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-header-box">
            <h3 class="section-header-title">⚙️ Sections C & D: Simulation & Manual Controls</h3>
            <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Control Panel</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2 = st.columns([1.2, 2.8])

    with ctrl_col1:
        mode_label = st.radio(
            "Operating Mode Selection",
            ["🎧 Manual Slider Controls", "⚡ Automatic Simulation Mode"],
            index=0 if st.session_state.env_sim_mode == "manual" else 1,
            horizontal=False,
            key="env_mode_radio_choice_v3",
        )
        new_mode = "manual" if "Manual" in mode_label else "auto"
        if new_mode != st.session_state.env_sim_mode:
            st.session_state.env_sim_mode = new_mode
            st.rerun()

    with ctrl_col2:
        if st.session_state.env_sim_mode == "auto":
            if st.session_state.env_is_auto_running:
                status_pill = "🟢 **RUNNING** — Generating periodic bounded telemetry drift"
            elif st.session_state.env_is_paused:
                status_pill = "⏸️ **PAUSED** — Simulation frozen at current values"
            else:
                status_pill = "⏹️ **STOPPED / READY** — Ready to start"
            st.info(f"Simulator State: {status_pill} | Elapsed Ticks: #{sim.tick_count}")
        else:
            st.info("🎧 **Manual Slider Mode Active:** Adjust parameter sliders or trigger scenario presets below.")

    # Automatic Simulation Controls
    if st.session_state.env_sim_mode == "auto":
        b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([1.2, 1, 1, 1, 2])

        with b_col1:
            if not st.session_state.env_is_auto_running:
                btn_label = "▶️ Resume" if st.session_state.env_is_paused else "▶️ Start"
                if st.button(btn_label, type="primary", use_container_width=True, key="btn_start_sim_v3"):
                    st.session_state.env_is_auto_running = True
                    st.session_state.env_is_paused = False
                    sim.start_auto_simulation()
                    st.rerun()
            else:
                st.button("⏳ Running...", disabled=True, use_container_width=True, key="btn_running_sim_v3")

        with b_col2:
            if st.session_state.env_is_auto_running:
                if st.button("⏸️ Pause", use_container_width=True, key="btn_pause_sim_v3"):
                    st.session_state.env_is_auto_running = False
                    st.session_state.env_is_paused = True
                    sim.pause_auto_simulation()
                    st.rerun()
            else:
                st.button("⏸️ Pause", disabled=True, use_container_width=True, key="btn_pause_disabled_v3")

        with b_col3:
            if st.button("⏭️ Step Once", use_container_width=True, key="btn_step_sim_v3"):
                readings = sim.step_simulation()
                for k, r in readings.items():
                    st.session_state[f"env_slider_{k}"] = r.value
                st.rerun()

        with b_col4:
            if st.button("🔄 Reset", use_container_width=True, key="btn_reset_auto_sim_v3"):
                st.session_state.env_is_auto_running = False
                st.session_state.env_is_paused = False
                defaults = sim.reset()
                for k, r in defaults.items():
                    st.session_state[f"env_slider_{k}"] = r.value
                st.success("Simulator restored to documented default baseline.")
                st.rerun()

        with b_col5:
            st.session_state.env_auto_interval = st.select_slider(
                "Update Interval (seconds)",
                options=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                value=st.session_state.env_auto_interval,
                key="select_auto_interval_v3",
            )

    # Manual Controls & Quick Scenario Presets
    else:
        st.markdown("##### Demonstration Scenario Presets:")
        p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
        with p_col1:
            if st.button("🟢 All Safe", use_container_width=True, key="btn_preset_safe_v3"):
                defaults = sim.reset()
                for k, r in defaults.items():
                    st.session_state[f"env_slider_{k}"] = r.value
                st.rerun()
        with p_col2:
            if st.button("🟠 Dust & Noise", use_container_width=True, key="btn_preset_warning_v3"):
                sim.set_reading("dust", 110.0)
                sim.set_reading("noise", 78.0)
                st.session_state["env_slider_dust"] = 110.0
                st.session_state["env_slider_noise"] = 78.0
                st.rerun()
        with p_col3:
            if st.button("🔴 Gas Leak Danger", use_container_width=True, key="btn_preset_gas_danger_v3"):
                sim.set_reading("gas", 65.0)
                st.session_state["env_slider_gas"] = 65.0
                st.rerun()
        with p_col4:
            if st.button("🔴 Heat Stress", use_container_width=True, key="btn_preset_thermal_danger_v3"):
                sim.set_reading("temperature", 42.0)
                sim.set_reading("humidity", 75.0)
                st.session_state["env_slider_temperature"] = 42.0
                st.session_state["env_slider_humidity"] = 75.0
                st.rerun()
        with p_col5:
            if st.button("🚨 Multi-Hazard", use_container_width=True, key="btn_preset_multi_danger_v3"):
                sim.set_reading("gas", 75.0)           # DANGER
                sim.set_reading("dust", 180.0)         # DANGER
                sim.set_reading("noise", 82.0)         # WARNING
                sim.set_reading("temperature", 34.0)   # WARNING
                st.session_state["env_slider_gas"] = 75.0
                st.session_state["env_slider_dust"] = 180.0
                st.session_state["env_slider_noise"] = 82.0
                st.session_state["env_slider_temperature"] = 34.0
                st.rerun()
        with p_col6:
            if st.button("🔄 Reset Defaults", use_container_width=True, key="btn_preset_reset_v3"):
                defaults = sim.reset()
                for k, r in defaults.items():
                    st.session_state[f"env_slider_{k}"] = r.value
                st.rerun()

        # Parameter Sliders utilizing configured ranges from config/environmental_thresholds.json
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            t_meta = sim.get_parameter_metadata("temperature")
            t_key = "env_slider_temperature"
            if t_key not in st.session_state:
                st.session_state[t_key] = float(t_meta["default_value"])
            temp_val = st.slider(
                f"🌡️ {t_meta['display_name']} ({t_meta['unit']})",
                min_value=float(t_meta["min_value"]),
                max_value=float(t_meta["max_value"]),
                step=float(t_meta["step"]),
                key=t_key,
                help=f"{t_meta.get('description', '')} Range: [{t_meta['min_value']}, {t_meta['max_value']} {t_meta['unit']}]",
            )
            sim.set_reading("temperature", temp_val)

        with s_col2:
            h_meta = sim.get_parameter_metadata("humidity")
            h_key = "env_slider_humidity"
            if h_key not in st.session_state:
                st.session_state[h_key] = float(h_meta["default_value"])
            hum_val = st.slider(
                f"💧 {h_meta['display_name']} ({h_meta['unit']})",
                min_value=float(h_meta["min_value"]),
                max_value=float(h_meta["max_value"]),
                step=float(h_meta["step"]),
                key=h_key,
                help=f"{h_meta.get('description', '')} Range: [{h_meta['min_value']}, {h_meta['max_value']} {h_meta['unit']}]",
            )
            sim.set_reading("humidity", hum_val)

        with s_col3:
            d_meta = sim.get_parameter_metadata("dust")
            d_key = "env_slider_dust"
            if d_key not in st.session_state:
                st.session_state[d_key] = float(d_meta["default_value"])
            dust_val = st.slider(
                f"🏗️ {d_meta['display_name']} ({d_meta['unit']})",
                min_value=float(d_meta["min_value"]),
                max_value=float(d_meta["max_value"]),
                step=float(d_meta["step"]),
                key=d_key,
                help=f"{d_meta.get('description', '')} Range: [{d_meta['min_value']}, {d_meta['max_value']} {d_meta['unit']}]",
            )
            sim.set_reading("dust", dust_val)

        s_col4, s_col5 = st.columns(2)
        with s_col4:
            n_meta = sim.get_parameter_metadata("noise")
            n_key = "env_slider_noise"
            if n_key not in st.session_state:
                st.session_state[n_key] = float(n_meta["default_value"])
            noise_val = st.slider(
                f"🔊 {n_meta['display_name']} ({n_meta['unit']})",
                min_value=float(n_meta["min_value"]),
                max_value=float(n_meta["max_value"]),
                step=float(n_meta["step"]),
                key=n_key,
                help=f"{n_meta.get('description', '')} Range: [{n_meta['min_value']}, {n_meta['max_value']} {n_meta['unit']}]",
            )
            sim.set_reading("noise", noise_val)

        with s_col5:
            g_meta = sim.get_parameter_metadata("gas")
            g_key = "env_slider_gas"
            if g_key not in st.session_state:
                st.session_state[g_key] = float(g_meta["default_value"])
            gas_val = st.slider(
                f"☣️ {g_meta['display_name']} ({g_meta['unit']})",
                min_value=float(g_meta["min_value"]),
                max_value=float(g_meta["max_value"]),
                step=float(g_meta["step"]),
                key=g_key,
                help=f"{g_meta.get('description', '')} Range: [{g_meta['min_value']}, {g_meta['max_value']} {g_meta['unit']}]",
            )
            sim.set_reading("gas", gas_val)

    # Re-evaluate site assessment with current slider values
    site_assessment = sim.assess_current_hazards()

    # -------------------------------------------------------------------------
    # Prioritized Active Hazards List
    # -------------------------------------------------------------------------
    render_active_hazard_prioritized_list(site_assessment)

    # -------------------------------------------------------------------------
    # Section B: Five Environmental Hazard Cards
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-header-box">
            <h3 class="section-header-title">📊 Section B: Five Environmental Parameter Cards</h3>
            <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Real-Time Status & Threshold Evaluation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_hazard_card_v3(site_assessment.all_assessments["temperature"], sim.get_parameter_metadata("temperature"))
    with c2:
        render_hazard_card_v3(site_assessment.all_assessments["humidity"], sim.get_parameter_metadata("humidity"))
    with c3:
        render_hazard_card_v3(site_assessment.all_assessments["dust"], sim.get_parameter_metadata("dust"))
    with c4:
        render_hazard_card_v3(site_assessment.all_assessments["noise"], sim.get_parameter_metadata("noise"))
    with c5:
        render_hazard_card_v3(site_assessment.all_assessments["gas"], sim.get_parameter_metadata("gas"))

    # Render Legend
    render_legend()

    # Configured Threshold Matrix Reference
    with st.expander("📋 Configured Thresholds & Occupational Hygiene Reference"):
        st.markdown(f"**Regulatory Notice:** {sim.disclaimer}")
        st.markdown("##### Matrix from `config/environmental_thresholds.json`")

        threshold_rows = []
        for key in sim.get_parameter_keys():
            m = sim.get_parameter_metadata(key)
            t = m["thresholds"]
            unit = m["unit"]
            if t["type"] == "range":
                safe_str = f"{t['safe_min']} – {t['safe_max']} {unit}"
                warning_str = f"Low: [{t['warning_min']} – {t['safe_min']}) OR High: ({t['safe_max']} – {t['warning_max']}] {unit}"
                danger_str = f"< {t['warning_min']} OR > {t['warning_max']} {unit}"
            else:
                safe_str = f"≤ {t['warning_threshold']} {unit}"
                warning_str = f"> {t['warning_threshold']} and ≤ {t['danger_threshold']} {unit}"
                danger_str = f"> {t['danger_threshold']} {unit}"

            threshold_rows.append({
                "Parameter": m["display_name"],
                "Unit": unit,
                "Configured Range": f"{m['min_value']} to {m['max_value']} {unit}",
                "🟢 Safe Range": safe_str,
                "🟠 Warning Threshold": warning_str,
                "🔴 Danger Threshold": danger_str,
            })

        st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Controlled Auto Simulation Loop
    # -------------------------------------------------------------------------
    if st.session_state.env_sim_mode == "auto" and st.session_state.env_is_auto_running:
        time.sleep(float(st.session_state.env_auto_interval))
        sim.step_simulation()
        # Keep sliders in sync
        for k, r in sim.get_all_readings().items():
            st.session_state[f"env_slider_{k}"] = r.value
        st.rerun()


def render_history_reports_view(sim: Optional[EnvironmentalSimulator] = None):
    """
    Renders the dedicated Hazard History and Report Generation View:
    - Section E: Time-series trend charts and filtered telemetry table
    - Section F: Hazard state-transition timeline & event log
    - Section G: Telemetry summary statistics
    - Section H: Compliance & audit report exports (CSV & JSON)
    """
    render_dashboard_styles()

    if sim is None:
        sim = get_or_create_simulator()

    st.markdown(
        f"""
        <div class="monitor-header">
            <h1 class="monitor-title">📊 SafeEdge Hazard History & Reports</h1>
            <p class="monitor-subtitle">
                Historical Telemetry Analysis, State-Transition Event Log, Summary Statistics, and Multi-Format Report Exports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sim-disclaimer-banner">
            <span>⚠️</span>
            <span><strong>{PHASE3_DISCLAIMER_TITLE}:</strong> {PHASE3_DISCLAIMER_SUB}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Section E: Hazard History & Telemetry Analysis
    # -------------------------------------------------------------------------
    render_history_section(sim)

    # -------------------------------------------------------------------------
    # Section F: Hazard State-Transition Timeline & Event Log
    # -------------------------------------------------------------------------
    render_event_log_section(sim)

    # -------------------------------------------------------------------------
    # Section G: Telemetry Summary Statistics
    # -------------------------------------------------------------------------
    render_summary_statistics_section(sim)

    # -------------------------------------------------------------------------
    # Section H: Compliance & Audit Report Export (CSV & JSON)
    # -------------------------------------------------------------------------
    render_report_export_section(sim)


def render_environmental_dashboard():
    """All-in-one rendering entry point combining simulator cockpit and history reports."""
    sim = get_or_create_simulator()
    render_simulator_view(sim)
    render_history_reports_view(sim)
