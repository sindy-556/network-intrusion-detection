"""Predictions page with improved functionality."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processor import DataProcessor


def show_predictions_page():
    """Display the predictions page.
    
    Shows single prediction interface for making predictions on individual connections.
    Requires a trained model to be available in session state.
    """
    st.title("🔮 Make Predictions")

    if not st.session_state.get('model_trained', False):
        st.warning("Please train a model first!")
        return

    st.markdown("### Predict on New Data")

    _show_single_prediction_tab()


def _show_single_prediction_tab():
    """Show single prediction interface.
    
    Creates dynamic input form based on trained model features
    and displays prediction results with confidence scores.
    """
    st.markdown("#### Enter Network Connection Features")

    # Get the feature names from the trained model
    feature_names = st.session_state.get('feature_names', [])
    
    if not feature_names:
        st.error("No feature information available. Please retrain the model.")
        return

    # Create input form with actual features
    with st.form("prediction_form"):
        # Store input values
        input_values = {}
        
        # Create columns for better layout
        cols = st.columns(3)
        
        # Get original column names before feature engineering
        original_columns = st.session_state.processor._get_default_column_names()[:-1]  # Remove 'difficulty'
        
        # Basic connection features that are commonly available
        basic_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in'
        ]
        
        col_idx = 0
        for i, feature in enumerate(basic_features):
            if feature in original_columns:
                with cols[col_idx % 3]:
                    input_values[feature] = _create_feature_input(feature)
                col_idx += 1
        
        # Additional features in expandable section
        with st.expander("Advanced Features (Optional)", expanded=False):
            advanced_features = [col for col in original_columns if col not in basic_features and col != 'labels']
            
            cols_adv = st.columns(3)
            for i, feature in enumerate(advanced_features):
                with cols_adv[i % 3]:
                    input_values[feature] = _create_feature_input(feature, advanced=True)

        submitted = st.form_submit_button("🎯 Predict", type="primary")

        if submitted:
            _make_single_prediction(input_values, feature_names)


def _create_feature_input(feature_name, advanced=False):
    """Create appropriate input widget for a feature.
    
    Args:
        feature_name (str): Name of the feature to create input for.
        advanced (bool): Whether this is an advanced feature with relaxed constraints.
    
    Returns:
        Any: Streamlit input widget value (selectbox, number_input, etc.).
    """
    
    # Get some context about the feature from the loaded data
    if st.session_state.get('data') is not None:
        sample_data = st.session_state.data
        feature_data = sample_data.get(feature_name)
        
        if feature_data is not None:
            # Determine appropriate input type based on data
            if feature_data.dtype == 'object':
                # Categorical feature
                unique_values = feature_data.unique()[:20]  # Limit to first 20 unique values
                return st.selectbox(
                    feature_name.replace('_', ' ').title(),
                    options=list(unique_values),
                    key=f"input_{feature_name}"
                )
            elif feature_data.nunique() == 2 and set(feature_data.unique()).issubset({0, 1}):
                # Binary feature
                return st.selectbox(
                    feature_name.replace('_', ' ').title(),
                    options=[0, 1],
                    key=f"input_{feature_name}"
                )
            else:
                # Numerical feature
                min_val = float(feature_data.min()) if not advanced else 0
                max_val = float(feature_data.max()) if not advanced else 10000
                default_val = float(feature_data.median()) if not advanced else min_val
                
                return st.number_input(
                    feature_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{feature_name}"
                )
    
    # Fallback for unknown features
    return st.number_input(
        feature_name.replace('_', ' ').title(),
        min_value=0,
        value=0,
        key=f"input_{feature_name}"
    )


def _make_single_prediction(input_values, feature_names):
    """Make a prediction on single input.
    
    Args:
        input_values (dict): Dictionary of feature names to values.
        feature_names (list): List of feature names used by the trained model.
    """
    try:
        # Create DataFrame from input values
        input_df = pd.DataFrame([input_values])
        
        # Apply the same preprocessing as during training
        processed_df = _preprocess_prediction_data(input_df)
        
        # Ensure we have the right features
        missing_features = [col for col in feature_names if col not in processed_df.columns]
        if missing_features:
            st.error(f"Feature mismatch. Missing features: {missing_features}")
            st.info(f"Available features: {list(processed_df.columns)}")
            st.info("Please retrain the model or check your inputs.")
            return
        
        # Select only the features used in training
        X_pred = processed_df[feature_names]
        
        # Make prediction
        model = st.session_state.model
        model_type = st.session_state.model_type
        
        if model_type == "multiclass":
            prediction = model.predict(X_pred)[0]
            probabilities = model.predict_proba(X_pred)[0] if hasattr(model.model, 'predict_proba') else None
            
            # Get class names
            target_names = st.session_state.get('target_names', [])
            if prediction < len(target_names):
                predicted_class = target_names[prediction]
            else:
                predicted_class = f"Class {prediction}"
            
            # Display results
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                if predicted_class.lower() == 'normal':
                    st.success(f"**Prediction: {predicted_class.upper()}**")
                else:
                    st.error(f"**Prediction: {predicted_class.upper()}**")
            
            with col2:
                if probabilities is not None:
                    confidence = probabilities[prediction] * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Show top 3 probabilities
                    top_3_idx = np.argsort(probabilities)[-3:][::-1]
                    st.markdown("**Top 3 Predictions:**")
                    for idx in top_3_idx:
                        class_name = target_names[idx] if idx < len(target_names) else f"Class {idx}"
                        st.write(f"• {class_name}: {probabilities[idx]*100:.1f}%")
        
        else:  # Two-phase detection
            binary_pred, multiclass_pred = model.predict(X_pred)
            
            st.markdown("### 🎯 Two-Phase Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Binary Classification")
                if binary_pred[0] == 1:  # Assuming 1 is normal
                    st.success("**Prediction: NORMAL**")
                else:
                    st.error("**Prediction: ATTACK**")
            
            with col2:
                st.markdown("#### 🔍 Attack Type Classification")
                if binary_pred[0] == 0:  # Attack detected
                    attack_type = multiclass_pred[0]
                    target_names = st.session_state.processor.get_label_names("labels5")
                    if attack_type < len(target_names):
                        attack_class = target_names[attack_type]
                        st.warning(f"**Attack Type: {attack_class.upper()}**")
                    else:
                        st.warning(f"**Attack Type: Unknown (Class {attack_type})**")
                else:
                    st.info("No attack detected")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Please check your input values and try again.")


def _preprocess_prediction_data(input_df):
    """Apply the same preprocessing as during training.
    
    Args:
        input_df (pd.DataFrame): Input data to preprocess.
    
    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction.
    """
    try:
        # Create a copy to avoid modifying original
        df = input_df.copy()
        
        # Add required columns for preprocessing (dummy values)
        df['labels'] = 'normal'
        if 'difficulty' not in df.columns:
            df['difficulty'] = 0  # Add dummy difficulty column
            
        # Ensure all expected columns are present with default values
        expected_columns = st.session_state.processor._get_default_column_names()
        for col in expected_columns:
            if col not in df.columns and col not in ['labels', 'difficulty']:
                # Add missing columns with appropriate default values
                if col in ['protocol_type', 'service', 'flag']:
                    df[col] = 'unknown'  # Default for categorical
                else:
                    df[col] = 0  # Default for numerical
        
        # Apply data processor transformations
        processor = st.session_state.processor
        df_processed = processor.create_target_labels(df)
        
        # Remove label columns but keep difficulty for feature engineering
        feature_df = df_processed.drop(columns=["labels", "labels2", "labels5"])
        
        # Apply the same feature engineering pipeline that was used during training
        fe_config = st.session_state.get('fe_config', {})
        if fe_config.get('apply_feature_engineering', False):
            # Use the saved feature engineering pipeline
            feature_engineer = st.session_state.get('feature_engineer')
            if feature_engineer is not None:
                # Transform using the fitted pipeline (this may remove difficulty)
                feature_df = feature_engineer.transform(feature_df)
            else:
                # Fallback: basic categorical encoding and remove difficulty
                categorical_cols = ['protocol_type', 'service', 'flag']
                for col in categorical_cols:
                    if col in feature_df.columns:
                        if feature_df[col].dtype == 'object':
                            feature_df[col] = pd.Categorical(feature_df[col]).codes
                # Remove difficulty in fallback case
                if "difficulty" in feature_df.columns:
                    feature_df = feature_df.drop(columns=["difficulty"])
        else:
            # If no feature engineering, remove difficulty column
            if "difficulty" in feature_df.columns:
                feature_df = feature_df.drop(columns=["difficulty"])
        
        return feature_df
        
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}")
        return input_df

