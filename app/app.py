"""Main Streamlit application for Network Intrusion Detection System - Refactored Version."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import page modules
from components.home import show_home_page, auto_load_data
from components.data_upload import show_data_upload_page
from components.eda import show_eda_page
from components.model_training import show_model_training_page
from components.predictions import show_predictions_page
from components.about import show_about_page

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    .status-indicator {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
def initialize_session_state():
    """Initialise session state variables.
    
    Sets up default values for all session state variables used
    throughout the application for data, models, and configuration.
    """
    default_states = {
        "data_loaded": False,
        "model_trained": False,
        "data": None,
        "train_data": None,
        "test_data": None,
        "processor": None,
        "model": None,
        "model_type": None,
        "X_test": None,
        "y_test": None,
        "feature_names": None,
        "fe_config": None,
        "training_config": None,
        "target_names": None,
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# Auto-load data on first run
if not st.session_state.get('auto_load_attempted', False):
    auto_load_data()
    st.session_state.auto_load_attempted = True

# Sidebar
with st.sidebar:
    st.title("🛡️ IDS Control Panel")

    # Navigation
    page = st.selectbox(
        "Navigation",
        [
            "Home",
            "Data Upload", 
            "EDA Dashboard",
            "Model Training",
            "Predictions",
            "About",
        ],
        key="navigation"
    )

    st.markdown("---")

    # Status indicators
    st.markdown("### 📊 System Status")
    
    # Data status
    if st.session_state.get('data_loaded', False):
        st.markdown(
            '<span class="status-indicator status-success">✓ Data Loaded</span>', 
            unsafe_allow_html=True
        )
        if st.session_state.get('train_data') is not None:
            st.caption(f"Training: {len(st.session_state.train_data):,} samples")
        if st.session_state.get('test_data') is not None:
            st.caption(f"Test: {len(st.session_state.test_data):,} samples")
    else:
        st.markdown(
            '<span class="status-indicator status-warning">⚠️ No Data</span>', 
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model status
    if st.session_state.get('model_trained', False):
        st.markdown(
            '<span class="status-indicator status-success">✓ Model Ready</span>', 
            unsafe_allow_html=True
        )
        if st.session_state.get('training_config'):
            config = st.session_state.training_config
            st.caption(f"Algorithm: {config.get('model_type', 'Unknown')}")
            st.caption(f"Type: {config.get('classification_type', 'Unknown')}")
    else:
        st.markdown(
            '<span class="status-indicator status-warning">⚠️ No Model</span>', 
            unsafe_allow_html=True
        )

    # Quick actions
    if st.session_state.get('data_loaded', False) and st.session_state.get('model_trained', False):
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Retrain Model", help="Go to model training"):
            st.session_state.navigation = "Model Training"
            st.rerun()
            
        if st.button("🔮 Make Prediction", help="Go to predictions"):
            st.session_state.navigation = "Predictions"
            st.rerun()

    # Footer info
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: centre; font-size: 0.8rem; color: #666;'>
            <p>🛡️ Network IDS v2.0</p>
            <p>Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Main content area
def render_page():
    """Render the selected page.
    
    Routes to the appropriate page component based on the current
    navigation selection in the sidebar.
    """
    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "EDA Dashboard":
        show_eda_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "About":
        show_about_page()
    else:
        st.error("Page not found!")

# Render the current page
render_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: centre'>
        <p style='color: #888; font-size: 0.9rem;'>
            Network Intrusion Detection System v2.0 | 
            Made with ❤️ using Streamlit | 
            <a href='https://github.com/yourusername/network-intrusion-detection' target='_blank'>View on GitHub</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)