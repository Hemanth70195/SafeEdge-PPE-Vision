"""
SafeEdge Environmental Hazard Monitor - Dedicated Application Entrypoint.
Phase 3: Professional construction-site safety dashboard suitable for demonstration.
Built with Streamlit.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.environmental_dashboard import render_environmental_dashboard

# Streamlit Page Configuration
st.set_page_config(
    page_title="SafeEdge Environmental Hazard Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if __name__ == "__main__":
    render_environmental_dashboard()
