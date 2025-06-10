"""Home page for Network Intrusion Detection System."""

import streamlit as st
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processor import DataProcessor


def show_home_page():
    """Display the home page.
    
    Shows the main dashboard with system overview, status indicators,
    and getting started instructions for the Network Intrusion Detection System.
    """
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("""
    Welcome to the Network Intrusion Detection System! This application uses machine learning
    to detect and classify network intrusions based on the NSL-KDD dataset.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
        <h3>🎯 Binary Detection</h3>
        <p>Classify network traffic as normal or attack</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
        <h3>🔍 Attack Classification</h3>
        <p>Identify specific attack types: DoS, Probe, R2L, U2R</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
        <h3>📊 Real-time Analysis</h3>
        <p>Interactive visualizations and performance metrics</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("### Getting Started")
    st.markdown("""
    1. **Data Loading**: NSL-KDD dataset auto-loads if available in data/ directory
    2. **Explore Data**: Visualise patterns and distributions in the EDA Dashboard
    3. **Train Model**: Choose models and customise feature engineering
    4. **Make Predictions**: Classify new network traffic
    """)

    # Show current data status
    if st.session_state.get('data_loaded', False):
        st.success("✅ Data is loaded and ready!")
        if st.session_state.get('train_data') is not None:
            st.info(f"Training data: {len(st.session_state.train_data):,} samples")
        if st.session_state.get('test_data') is not None:
            st.info(f"Test data: {len(st.session_state.test_data):,} samples")
    else:
        st.warning("⚠️ No data loaded. Check the Data Upload page.")


def auto_load_data():
    """Auto-load NSL-KDD dataset if available.
    
    Automatically detects and loads NSL-KDD training and test datasets
    from the data/ directory if they exist. Updates session state
    with loaded data and processor instance.
    
    Returns:
        None: Updates session state directly.
    """
    TRAIN_FILE = "data/KDDTrain+.csv"
    TEST_FILE = "data/KDDTest+.csv"
    
    # Check if data is already loaded
    if st.session_state.get('data_loaded', False):
        return
    
    # Check if local data is available
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        try:
            with st.spinner("Auto-loading NSL-KDD dataset..."):
                processor = DataProcessor()
                
                # Load training data
                train_data = processor.load_data(TRAIN_FILE)
                st.session_state.train_data = train_data
                
                # Load test data
                test_data = processor.load_data(TEST_FILE)
                st.session_state.test_data = test_data
                
                # Set primary data to training data
                st.session_state.data = train_data
                st.session_state.processor = processor
                st.session_state.data_loaded = True
                
                st.success(f"✅ Auto-loaded {len(train_data):,} training and {len(test_data):,} test samples")
                
        except Exception as e:
            st.error(f"❌ Failed to auto-load data: {str(e)}")
            st.info("Please use the Data Upload page to load data manually.")
    elif os.path.exists(TRAIN_FILE) or os.path.exists(TEST_FILE):
        # Only one file exists
        try:
            with st.spinner("Auto-loading available NSL-KDD data..."):
                processor = DataProcessor()
                
                if os.path.exists(TRAIN_FILE):
                    data = processor.load_data(TRAIN_FILE)
                    st.session_state.train_data = data
                    st.session_state.data = data
                    st.info(f"✅ Auto-loaded training data: {len(data):,} samples")
                else:
                    data = processor.load_data(TEST_FILE)
                    st.session_state.test_data = data
                    st.session_state.data = data
                    st.info(f"✅ Auto-loaded test data: {len(data):,} samples")
                
                st.session_state.processor = processor
                st.session_state.data_loaded = True
                
        except Exception as e:
            st.error(f"❌ Failed to auto-load data: {str(e)}")