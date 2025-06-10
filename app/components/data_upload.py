"""Data upload and management page."""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processor import DataProcessor
from src.utils import validate_data


def show_data_upload_page():
    """Display the data upload page.
    
    Shows data management interface with tabs for loading local NSL-KDD data,
    uploading custom datasets, and viewing data information.
    """
    st.title("📁 Data Management")

    tab1, tab2, tab3 = st.tabs(["Load Local Data", "Upload Custom Data", "Data Info"])

    with tab1:
        _show_local_data_tab()

    with tab2:
        _show_upload_tab()

    with tab3:
        _show_data_info_tab()


def _show_local_data_tab():
    """Show local data loading options.
    
    Displays interface for loading NSL-KDD datasets from the local data/ directory.
    Allows loading training data, test data, or both datasets.
    """
    st.markdown("### Load NSL-KDD Dataset")
    
    TRAIN_FILE = "data/KDDTrain+.csv"
    TEST_FILE = "data/KDDTest+.csv"
    local_data_available = os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)

    if local_data_available:
        st.success("✓ Local NSL-KDD dataset found in data/ directory")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "Load Training Data",
                type="primary",
                disabled=st.session_state.get('data_loaded', False),
            ):
                with st.spinner("Loading training data..."):
                    processor = DataProcessor()
                    df = processor.load_data(TRAIN_FILE)
                    st.session_state.train_data = df
                    st.session_state.data = df
                    st.session_state.processor = processor
                    st.session_state.data_loaded = True
                    st.success(f"✓ Loaded {len(df):,} training samples")
                    st.rerun()

        with col2:
            if st.button("Load Test Data", disabled=st.session_state.get('data_loaded', False)):
                with st.spinner("Loading test data..."):
                    processor = DataProcessor()
                    df = processor.load_data(TEST_FILE)
                    st.session_state.test_data = df
                    st.session_state.data = df
                    st.session_state.processor = processor
                    st.session_state.data_loaded = True
                    st.success(f"✓ Loaded {len(df):,} test samples")
                    st.rerun()

        with col3:
            if st.button("Load Both", disabled=st.session_state.get('data_loaded', False)):
                with st.spinner("Loading both datasets..."):
                    processor = DataProcessor()
                    train_df = processor.load_data(TRAIN_FILE)
                    test_df = processor.load_data(TEST_FILE)
                    st.session_state.train_data = train_df
                    st.session_state.test_data = test_df
                    st.session_state.data = train_df  # Use train data as primary
                    st.session_state.processor = processor
                    st.session_state.data_loaded = True
                    st.success(
                        f"✓ Loaded {len(train_df):,} training and {len(test_df):,} test samples"
                    )
                    st.rerun()

        if st.session_state.get('data_loaded', False):
            st.info(
                f"Currently using: {'Training' if st.session_state.data is st.session_state.get('train_data') else 'Test'} dataset"
            )

            # Option to switch between datasets
            if (
                st.session_state.get('train_data') is not None
                and st.session_state.get('test_data') is not None
            ):
                dataset_choice = st.radio(
                    "Switch dataset:",
                    ["Training", "Test"],
                    index=0
                    if st.session_state.data is st.session_state.get('train_data')
                    else 1,
                )
                if st.button("Switch"):
                    st.session_state.data = (
                        st.session_state.train_data
                        if dataset_choice == "Training"
                        else st.session_state.test_data
                    )
                    st.rerun()

            # Data preview
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.data.head())

            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{st.session_state.data.shape[0]:,}")
            with col2:
                st.metric("Columns", st.session_state.data.shape[1])
            with col3:
                st.metric("Attack Types", st.session_state.data["labels"].nunique())
            with col4:
                st.metric(
                    "Memory",
                    f"{st.session_state.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                )
                
        # Reset data button
        if st.session_state.get('data_loaded', False):
            if st.button("🗑️ Reset Data", type="secondary"):
                st.session_state.data_loaded = False
                st.session_state.data = None
                st.session_state.train_data = None
                st.session_state.test_data = None
                st.session_state.processor = None
                st.session_state.model_trained = False
                st.session_state.model = None
                st.success("Data reset successfully!")
                st.rerun()
                
    else:
        st.error("❌ No local data found in data/ directory")
        st.info("""
        Please ensure you have the following files in your data/ directory:
        - KDDTrain+.csv
        - KDDTest+.csv
        
        You can download the NSL-KDD dataset from:
        https://www.unb.ca/cic/datasets/nsl.html
        """)


def _show_upload_tab():
    """Show custom data upload options.
    
    Displays file upload interface for custom NSL-KDD format datasets.
    Validates uploaded data against expected format.
    """
    st.markdown("### Upload Custom Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a custom NSL-KDD format dataset",
    )

    if uploaded_file is not None:
        try:
            processor = DataProcessor()
            df = processor.load_data(uploaded_file)

            # Validate data
            is_valid, message = validate_data(
                df, processor._get_default_column_names()
            )

            if is_valid:
                st.session_state.data = df
                st.session_state.data_loaded = True
                st.session_state.processor = processor
                st.success(f"✓ {message}")

                # Show data preview
                st.markdown("### Data Preview")
                st.dataframe(df.head())

                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Attack Types", df["labels"].nunique())
                with col4:
                    st.metric(
                        "Memory",
                        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    )
            else:
                st.error(f"✗ {message}")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def _show_data_info_tab():
    """Show detailed data information.
    
    Displays comprehensive information about the loaded dataset including
    feature types, target distribution, and basic statistics.
    """
    if st.session_state.get('data_loaded', False):
        st.markdown("### Dataset Information")

        df = st.session_state.data

        # Data types
        st.markdown("#### Feature Types")
        feature_types = st.session_state.processor.get_feature_types(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Binary Features**: {len(feature_types['binary'])}")
            with st.expander("View Binary Features"):
                st.write(feature_types["binary"])
        with col2:
            st.info(
                f"**Categorical Features**: {len(feature_types['categorical'])}"
            )
            with st.expander("View Categorical Features"):
                st.write(feature_types["categorical"])
        with col3:
            st.info(f"**Continuous Features**: {len(feature_types['continuous'])}")
            with st.expander("View Continuous Features"):
                st.write(feature_types["continuous"])

        # Target distribution
        st.markdown("#### Target Distribution")
        attack_counts = df["labels"].value_counts()
        st.bar_chart(attack_counts)
    else:
        st.info("Please upload data first to view information.")