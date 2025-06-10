"""Model training page with customisable feature engineering."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.feature_engineer import FeatureEngineer
from src.models import IntrusionDetectionModel, TwoPhaseDetector
from src.utils import format_metrics
from src.visualiser import Visualiser


def show_model_training_page():
    """Display the model training page.
    
    This function renders the complete model training interface including
    feature engineering configuration and model selection options.
    """
    st.title("🤖 Model Training")

    if not st.session_state.get("data_loaded", False):
        st.warning("Please upload data first!")
        return

    df = st.session_state.data
    processor = st.session_state.processor

    # Process data
    df_processed = processor.create_target_labels(df)

    # Feature engineering configuration
    st.markdown("### 🔧 Feature Engineering Configuration")

    with st.expander("Feature Engineering Options", expanded=True):
        fe_config = _show_feature_engineering_options(df_processed)

    # Model configuration
    st.markdown("### 🎯 Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Select Model",
            [
                "random_forest",
                "decision_tree",
                "logistic_regression",
                "balanced_forest",
                "balanced_tree",
                "balanced_logistic",
            ],
            help="Choose the machine learning algorithm",
        )

        test_size = st.slider(
            "Test Set Size",
            0.1,
            0.5,
            0.3,
            0.05,
            help="Proportion of data to use for testing",
        )

    with col2:
        classification_type = st.radio(
            "Classification Type",
            ["Multi-class (5 classes)", "Two-phase (Binary + Multi-class)"],
            help="Choose between direct multi-class or two-phase detection",
        )

        random_state = st.number_input(
            "Random State",
            min_value=1,
            max_value=9999,
            value=42,
            help="Set random state for reproducible results",
        )

    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        _train_model(
            df_processed,
            fe_config,
            model_type,
            classification_type,
            test_size,
            random_state,
        )


def _show_feature_engineering_options(df_processed):
    """Show customisable feature engineering options.
    
    Args:
        df_processed: Processed DataFrame with target labels.
        
    Returns:
        dict: Configuration dictionary with user-selected feature engineering options.
    """
    config = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏗️ Feature Selection")

        config["apply_feature_engineering"] = st.checkbox(
            "Enable Feature Engineering",
            value=True,
            help="Apply automated feature selection and engineering",
        )

        if config["apply_feature_engineering"]:
            config["remove_constant"] = st.checkbox(
                "Remove Constant Features",
                value=True,
                help="Remove features with constant or near-constant values",
            )

            config["constant_threshold"] = st.slider(
                "Constant Threshold",
                0.95,
                1.0,
                0.998,
                0.001,
                help="Threshold for considering features as constant",
                disabled=not config["remove_constant"],
            )

            config["remove_duplicates"] = st.checkbox(
                "Remove Duplicate Features",
                value=True,
                help="Remove features that are exact duplicates",
            )

            config["remove_correlated"] = st.checkbox(
                "Remove Correlated Features",
                value=True,
                help="Remove highly correlated features",
            )

            config["correlation_threshold"] = st.slider(
                "Correlation Threshold",
                0.7,
                0.99,
                0.95,
                0.01,
                help="Threshold for removing correlated features",
                disabled=not config["remove_correlated"],
            )

    with col2:
        st.markdown("#### 🔄 Data Preprocessing")

        config["apply_scaling"] = st.checkbox(
            "Scale Continuous Features",
            value=True,
            help="Apply standard scaling to continuous features",
        )

        config["scaling_method"] = st.selectbox(
            "Scaling Method",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"],
            help="Choose scaling method for continuous features",
            disabled=not config["apply_scaling"],
        )

        config["encode_rare_labels"] = st.checkbox(
            "Encode Rare Labels",
            value=True,
            help="Group rare categorical values together",
        )

        config["rare_threshold"] = st.slider(
            "Rare Label Threshold",
            0.001,
            0.1,
            0.01,
            0.001,
            help="Threshold for considering labels as rare",
            disabled=not config["encode_rare_labels"],
        )

        config["handle_categorical"] = st.selectbox(
            "Categorical Encoding",
            ["ordinal", "onehot", "target"],
            help="Method for encoding categorical variables",
        )

    # Feature selection method
    if config.get("apply_feature_engineering", False):
        st.markdown("#### ⏩ Advanced Options")

        col3, col4 = st.columns(2)

        with col3:
            config["feature_selection_method"] = st.selectbox(
                "Feature Selection Method",
                ["variance", "cardinality", "correlation"],
                help="Method for selecting features in correlated feature removal",
            )

        with col4:
            config["max_features"] = st.number_input(
                "Max Features (0 = no limit)",
                min_value=0,
                max_value=1000,
                value=0,
                help="Maximum number of features to keep (0 for no limit)",
            )

    return config


def _train_model(
    df_processed, fe_config, model_type, classification_type, test_size, random_state
):
    """Train the model with the given configuration.
    
    Args:
        df_processed: Processed DataFrame with target labels.
        fe_config (dict): Feature engineering configuration options.
        model_type (str): Type of machine learning model to train.
        classification_type (str): Type of classification (multi-class or two-phase).
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random state for reproducible results.
    """

    with st.spinner("Training model..."):
        try:
            # Prepare data
            X = df_processed.drop(columns=["labels", "labels2", "labels5"])
            y = df_processed[["labels2", "labels5"]]

            # Split data
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Apply feature engineering if enabled
            if fe_config.get("apply_feature_engineering", False):
                # Create custom feature engineer with user settings
                engineer = _create_custom_feature_engineer(fe_config)
                X_train, X_test = engineer.fit_transform(X_train, X_test)

                # Show feature engineering results
                feature_info = engineer.get_feature_info()
                _display_feature_engineering_results(feature_info, X_train.shape[1])
                
                # Save the fitted feature engineer for predictions
                st.session_state.feature_engineer = engineer
            else:
                # Clear feature engineer if not using feature engineering
                st.session_state.feature_engineer = None

            # Train model based on classification type
            if classification_type == "Multi-class (5 classes)":
                model = IntrusionDetectionModel(model_type)
                model.train(X_train, y_train["labels5"])

                # Evaluate
                metrics = model.evaluate(X_test, y_test["labels5"])
                cm = model.get_confusion_matrix(X_test, y_test["labels5"])

                # Store in session state
                st.session_state.model = model
                st.session_state.model_type = "multiclass"
                st.session_state.target_names = (
                    st.session_state.processor.get_label_names("labels5")
                )

            else:  # Two-phase
                model = TwoPhaseDetector(model_type, model_type)
                model.train(X_train, y_train["labels2"], y_train["labels5"])

                # Evaluate
                metrics = model.evaluate(X_test, y_train["labels2"], y_test["labels5"])

                # Store in session state
                st.session_state.model = model
                st.session_state.model_type = "twophase"

            # Store additional info
            st.session_state.model_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = list(X_train.columns)
            st.session_state.fe_config = fe_config
            st.session_state.training_config = {
                "model_type": model_type,
                "classification_type": classification_type,
                "test_size": test_size,
                "random_state": random_state,
            }

            # Display results
            st.success("✅ Model trained successfully!")
            _display_training_results(metrics, model, classification_type)

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.info("Please check your feature engineering settings and try again.")


def _create_custom_feature_engineer(config):
    """Create a customised feature engineer based on user configuration.
    
    Args:
        config (dict): User configuration dictionary containing feature engineering options.
        
    Returns:
        CustomFeatureEngineer: Configured feature engineering instance.
    """
    from feature_engine.selection import (
        DropConstantFeatures,
        DropDuplicateFeatures,
        SmartCorrelatedSelection,
    )
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    from src.feature_engineer import RareLabelEncoder

    class CustomFeatureEngineer(FeatureEngineer):
        def __init__(self, config):
            super().__init__()
            self.config = config

            # Configure scaler based on user choice
            if config.get("scaling_method") == "MinMaxScaler":
                self.scaler = MinMaxScaler()
            elif config.get("scaling_method") == "RobustScaler":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()

            # Configure rare label encoder
            if config.get("encode_rare_labels", True):
                self.rare_encoder = RareLabelEncoder(
                    columns=["service"], threshold=config.get("rare_threshold", 0.01)
                )
            else:
                self.rare_encoder = None

        def create_feature_pipeline(self):
            """Create customised feature selection pipeline.
            
            Returns:
                Pipeline or None: Feature selection pipeline if steps are configured, None otherwise.
            """
            steps = []

            if self.config.get("remove_constant", True):
                steps.append(
                    (
                        "constant",
                        DropConstantFeatures(
                            tol=self.config.get("constant_threshold", 0.998)
                        ),
                    )
                )

            if self.config.get("remove_duplicates", True):
                steps.append(("duplicated", DropDuplicateFeatures()))

            if self.config.get("remove_correlated", True):
                steps.append(
                    (
                        "correlated",
                        SmartCorrelatedSelection(
                            threshold=self.config.get("correlation_threshold", 0.95),
                            selection_method=self.config.get(
                                "feature_selection_method", "variance"
                            ),
                        ),
                    )
                )

            from sklearn.pipeline import Pipeline

            return Pipeline(steps) if steps else None

        def fit_transform(self, X_train, X_test):
            """Custom fit and transform with user settings.
            
            Args:
                X_train: Training feature data.
                X_test: Test feature data.
                
            Returns:
                tuple: Transformed (X_train, X_test) data.
            """
            # Apply rare label encoding if enabled
            if self.rare_encoder:
                X_train = self.rare_encoder.fit_transform(X_train)
                X_test = self.rare_encoder.transform(X_test)

            # Identify feature types
            binary_features = self._get_binary_features(X_train)
            self.continuous_features = [
                col
                for col in X_train.columns
                if X_train[col].dtype in ["int64", "float64"]
                and col not in binary_features
                and col not in self.categorical_features
            ]

            # Scale continuous features if enabled
            if self.config.get("apply_scaling", True) and self.continuous_features:
                X_train[self.continuous_features] = self.scaler.fit_transform(
                    X_train[self.continuous_features]
                )
                X_test[self.continuous_features] = self.scaler.transform(
                    X_test[self.continuous_features]
                )

            # Apply feature selection if enabled
            self.feature_selector = self.create_feature_pipeline()
            if self.feature_selector:
                X_train = self.feature_selector.fit_transform(X_train)
                X_test = self.feature_selector.transform(X_test)

            # Handle categorical encoding
            categorical_in_train = [
                col for col in self.categorical_features if col in X_train.columns
            ]

            if categorical_in_train:
                encoding_method = self.config.get("handle_categorical", "ordinal")

                if encoding_method == "ordinal":
                    from feature_engine.encoding import OrdinalEncoder

                    self.ordinal_encoder = OrdinalEncoder(
                        encoding_method="arbitrary", variables=categorical_in_train
                    )
                    X_train = self.ordinal_encoder.fit_transform(X_train)
                    X_test = self.ordinal_encoder.transform(X_test)

                elif encoding_method == "onehot":
                    X_train = pd.get_dummies(X_train, columns=categorical_in_train)
                    X_test = pd.get_dummies(X_test, columns=categorical_in_train)
                    # Align columns
                    X_train, X_test = X_train.align(
                        X_test, join="left", axis=1, fill_value=0
                    )

            # Apply max features limit if specified
            max_features = self.config.get("max_features", 0)
            if max_features > 0 and X_train.shape[1] > max_features:
                # Select top features by variance
                feature_variances = X_train.var().sort_values(ascending=False)
                top_features = feature_variances.head(max_features).index
                X_train = X_train[top_features]
                X_test = X_test[top_features]

            return X_train, X_test

    return CustomFeatureEngineer(config)


def _display_feature_engineering_results(feature_info, final_feature_count):
    """Display feature engineering results.
    
    Args:
        feature_info (dict): Dictionary containing information about removed features.
        final_feature_count (int): Final number of features after engineering.
    """
    st.markdown("### 🔧 Feature Engineering Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Constant Features Removed",
            len(feature_info.get("constant_features", [])),
            help="Features with constant or near-constant values",
        )
    with col2:
        st.metric(
            "Duplicate Features Removed",
            len(feature_info.get("duplicate_features", [])),
            help="Features that are exact duplicates",
        )
    with col3:
        st.metric(
            "Correlated Features Removed",
            len(feature_info.get("correlated_features", [])),
            help="Highly correlated features",
        )
    with col4:
        st.metric(
            "Final Feature Count",
            final_feature_count,
            help="Number of features after engineering",
        )


def _display_training_results(metrics, model, classification_type):
    """Display model training results.
    
    Args:
        metrics (dict): Model performance metrics.
        model: Trained model instance.
        classification_type (str): Type of classification performed.
    """
    st.markdown("### 📊 Model Performance")

    if classification_type == "Multi-class (5 classes)":
        # Display metrics
        metrics_df = format_metrics(metrics)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📈 Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)

        with col2:
            # Feature importance for tree-based models
            if hasattr(model.model, "feature_importances_"):
                importance = model.get_feature_importance()
                if importance is not None and len(
                    st.session_state.feature_names
                ) == len(importance):
                    st.markdown("#### 🎯 Top 20 Feature Importances")
                    visualiser = Visualiser()
                    fig = visualiser.plot_feature_importance(
                        importance, st.session_state.feature_names
                    )
                    # Ensure consistent height with the metrics table
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### 🎯 Feature Importance")
                st.info("Feature importance not available for this model type")

        # Confusion matrix
        st.markdown("### 🎯 Confusion Matrix")
        cm = model.get_confusion_matrix(
            st.session_state.X_test, st.session_state.y_test["labels5"]
        )
        visualiser = Visualiser()
        fig = visualiser.plot_confusion_matrix(cm, st.session_state.target_names)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Two-phase
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Binary Classification Metrics")
            if metrics.get("binary"):
                binary_metrics_df = format_metrics(metrics["binary"])
                st.dataframe(binary_metrics_df, use_container_width=True)

        with col2:
            st.markdown("#### 🔍 Multi-class Classification Metrics")
            if metrics.get("multiclass"):
                multiclass_metrics_df = format_metrics(metrics["multiclass"])
                st.dataframe(multiclass_metrics_df, use_container_width=True)
            else:
                st.info("No attack samples in test set for multi-class evaluation")
