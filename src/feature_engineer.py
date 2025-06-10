"""Feature engineering module."""

from typing import Dict, List, Optional

import pandas as pd
from feature_engine.encoding import OrdinalEncoder
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode rare categories as 'rare'.
    
    A custom transformer that identifies and encodes rare categorical values
    based on frequency thresholds to improve model generalisation.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 0.01,
        replace_with: str = "rare",
    ):
        self.columns = columns
        self.threshold = threshold
        self.replace_with = replace_with
        self.rare_categories = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder to identify rare categories.
        
        Args:
            X (pd.DataFrame): Input dataframe with categorical features.
            y: Ignored, present for sklearn compatibility.
        
        Returns:
            RareLabelEncoder: Returns self for method chaining.
        """
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "category"]).columns

        for col in self.columns:
            if col in X.columns:
                frequencies = X[col].value_counts(normalize=True)
                self.rare_categories[col] = frequencies[
                    frequencies < self.threshold
                ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform rare categories to 'rare'.
        
        Args:
            X (pd.DataFrame): Input dataframe to transform.
        
        Returns:
            pd.DataFrame: Transformed dataframe with rare categories replaced.
        """
        X_encoded = X.copy()
        for col in self.columns:
            if col in X_encoded.columns and col in self.rare_categories:
                X_encoded[col] = X_encoded[col].apply(
                    lambda x: self.replace_with if x in self.rare_categories[col] else x
                )
        return X_encoded


class FeatureEngineer:
    """Handles feature engineering pipeline.
    
    Comprehensive feature engineering class that provides preprocessing,
    feature selection, and transformation capabilities for network intrusion
    detection data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.rare_encoder = RareLabelEncoder(columns=["service"])
        self.ordinal_encoder = None
        self.feature_selector = None
        self.continuous_features = []
        self.categorical_features = ["protocol_type", "service", "flag"]

    def create_feature_pipeline(self) -> Pipeline:
        """Create feature selection pipeline.
        
        Creates a pipeline that removes constant features, duplicate features,
        and highly correlated features to improve model performance.
        
        Returns:
            Pipeline: Configured feature selection pipeline.
        """
        return Pipeline(
            [
                ("constant", DropConstantFeatures(tol=0.998)),
                ("duplicated", DropDuplicateFeatures()),
                ("correlated", SmartCorrelatedSelection(selection_method="variance")),
            ]
        )

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Fit and transform features.
        
        Applies the complete feature engineering pipeline including rare label
        encoding, scaling, feature selection, and categorical encoding.
        
        Args:
            X_train (pd.DataFrame): Training feature dataframe.
            X_test (pd.DataFrame): Test feature dataframe.
        
        Returns:
            tuple: Transformed (X_train, X_test) dataframes.
        """
        # Apply rare label encoding
        X_train = self.rare_encoder.fit_transform(X_train)
        X_test = self.rare_encoder.transform(X_test)

        # Identify continuous features (excluding binary)
        binary_features = self._get_binary_features(X_train)
        self.continuous_features = [
            col
            for col in X_train.columns
            if X_train[col].dtype in ["int64", "float64"]
            and col not in binary_features
            and col not in self.categorical_features
        ]

        # Scale continuous features
        if self.continuous_features:
            X_train[self.continuous_features] = self.scaler.fit_transform(
                X_train[self.continuous_features]
            )
            X_test[self.continuous_features] = self.scaler.transform(
                X_test[self.continuous_features]
            )

        # Apply feature selection
        self.feature_selector = self.create_feature_pipeline()
        X_train = self.feature_selector.fit_transform(X_train)
        X_test = self.feature_selector.transform(X_test)

        # Encode categorical features
        categorical_in_train = [
            col for col in self.categorical_features if col in X_train.columns
        ]
        if categorical_in_train:
            self.ordinal_encoder = OrdinalEncoder(
                encoding_method="arbitrary", variables=categorical_in_train
            )
            X_train = self.ordinal_encoder.fit_transform(X_train)
            X_test = self.ordinal_encoder.transform(X_test)

        return X_train, X_test

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers.
        
        Applies all fitted transformations (rare encoding, scaling, feature selection,
        categorical encoding) to new data for prediction.
        
        Args:
            X (pd.DataFrame): Input dataframe to transform.
        
        Returns:
            pd.DataFrame: Transformed dataframe ready for model prediction.
        """
        X_transformed = X.copy()
        
        # Apply rare label encoding
        if hasattr(self.rare_encoder, 'rare_labels_'):
            X_transformed = self.rare_encoder.transform(X_transformed)
        
        # Scale continuous features
        if self.continuous_features and hasattr(self.scaler, 'mean_'):
            # Only scale features that exist in the input
            available_continuous = [col for col in self.continuous_features if col in X_transformed.columns]
            if available_continuous:
                X_transformed[available_continuous] = self.scaler.transform(
                    X_transformed[available_continuous]
                )
        
        # Apply feature selection with error handling for missing features
        if self.feature_selector is not None:
            try:
                X_transformed = self.feature_selector.transform(X_transformed)
            except ValueError as e:
                if "feature names" in str(e) and "missing" in str(e):
                    # Get the features that the selector was trained on
                    if hasattr(self.feature_selector, 'feature_names_in_'):
                        expected_features = self.feature_selector.feature_names_in_
                        # Add missing features with zeros
                        for feature in expected_features:
                            if feature not in X_transformed.columns:
                                X_transformed[feature] = 0
                        # Reorder columns to match training order
                        X_transformed = X_transformed[expected_features]
                        # Try transform again
                        X_transformed = self.feature_selector.transform(X_transformed)
                    else:
                        raise e
                else:
                    raise e
        
        # Encode categorical features
        if self.ordinal_encoder is not None and hasattr(self.ordinal_encoder, 'encoder_dict_'):
            X_transformed = self.ordinal_encoder.transform(X_transformed)
        
        return X_transformed

    def _get_binary_features(self, df: pd.DataFrame) -> List[str]:
        """Identify binary features.
        
        Args:
            df (pd.DataFrame): Input dataframe to analyse.
        
        Returns:
            List[str]: List of column names that contain only binary values (0, 1).
        """
        return [
            col
            for col in df.columns
            if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1})
        ]

    def get_feature_info(self) -> Dict[str, List[str]]:
        """Get information about selected/dropped features.
        
        Returns:
            Dict[str, List[str]]: Dictionary containing lists of features that
                were dropped by each step in the feature selection pipeline.
        """
        info = {}
        if self.feature_selector:
            info["constant_features"] = getattr(
                self.feature_selector.named_steps["constant"], "features_to_drop_", []
            )
            info["duplicate_features"] = getattr(
                self.feature_selector.named_steps["duplicated"], "features_to_drop_", []
            )
            info["correlated_features"] = getattr(
                self.feature_selector.named_steps["correlated"], "features_to_drop_", []
            )
        return info
