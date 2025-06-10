"""Machine learning models for intrusion detection."""

from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


class IntrusionDetectionModel:
    """Base class for intrusion detection models.
    
    Provides a unified interface for training and evaluating various machine
    learning models for network intrusion detection tasks.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """Create the specified model.
        
        Returns:
            sklearn estimator: Configured machine learning model instance.
        
        Raises:
            ValueError: If the specified model type is not supported.
        """
        models = {
            "logistic_regression": LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=1000
            ),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "balanced_logistic": LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
            ),
            "balanced_tree": DecisionTreeClassifier(
                class_weight="balanced", random_state=42
            ),
            "balanced_forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            ),
        }

        if self.model_type not in models:
            raise ValueError(f"Model type {self.model_type} not supported")

        return models[self.model_type]

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predicted class labels.
        
        Raises:
            ValueError: If model has not been trained yet.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Class probabilities if supported by the model, 
                None otherwise.
        
        Raises:
            ValueError: If model has not been trained yet.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target labels.
        
        Returns:
            Dict[str, float]: Dictionary containing various performance metrics
                including accuracy, precision, recall, F1-score, Cohen's kappa,
                Hamming loss, and Matthews correlation coefficient.
        """
        y_pred = self.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "cohen_kappa": cohen_kappa_score(y_test, y_pred),
            "hamming_loss": hamming_loss(y_test, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        }

        return metrics

    def get_confusion_matrix(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> np.ndarray:
        """Get confusion matrix.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target labels.
        
        Returns:
            np.ndarray: Confusion matrix as a 2D array.
        """
        y_pred = self.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(
        self, X_test: pd.DataFrame, y_test: pd.Series, target_names: List[str] = None
    ) -> str:
        """Get detailed classification report.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target labels.
            target_names (List[str], optional): Display names for the classes.
                Defaults to None.
        
        Returns:
            str: Formatted classification report with precision, recall,
                and F1-score for each class.
        """
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred, target_names=target_names)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances (for tree-based models).
        
        Returns:
            pd.DataFrame: Feature importances if the model supports them,
                None otherwise.
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def save_model(self, filepath: str):
        """Save model to file.
        
        Args:
            filepath (str): Path where the model should be saved.
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str):
        """Load model from file.
        
        Args:
            filepath (str): Path to the saved model file.
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True


class TwoPhaseDetector:
    """Two-phase detection: binary classification followed by multi-class.
    
    Implements a hierarchical classification approach where binary classification
    is performed first to distinguish normal from attack traffic, followed by
    multi-class classification to categorise attack types.
    """

    def __init__(
        self,
        binary_model_type: str = "random_forest",
        multiclass_model_type: str = "random_forest",
    ):
        self.binary_model = IntrusionDetectionModel(binary_model_type)
        self.multiclass_model = IntrusionDetectionModel(multiclass_model_type)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train_binary: pd.Series,
        y_train_multiclass: pd.Series,
    ):
        """Train both models.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train_binary (pd.Series): Binary labels (normal vs attack).
            y_train_multiclass (pd.Series): Multi-class labels (attack types).
        """
        # Train binary classifier
        self.binary_model.train(X_train, y_train_binary)

        # Train multi-class classifier on attack samples only
        y_train_pred = self.binary_model.predict(X_train)
        attack_mask = y_train_pred == 0  # Assuming 0 is 'attack'

        if attack_mask.sum() > 0:
            X_train_attacks = X_train[attack_mask]
            y_train_attacks = y_train_multiclass[attack_mask]
            self.multiclass_model.train(X_train_attacks, y_train_attacks)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make two-phase predictions.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Binary predictions and multi-class
                predictions. Multi-class predictions are -1 for normal traffic.
        """
        # Phase 1: Binary classification
        binary_pred = self.binary_model.predict(X)

        # Phase 2: Multi-class for detected attacks
        multiclass_pred = np.full(len(X), -1)  # -1 for normal traffic
        attack_mask = binary_pred == 0

        if attack_mask.sum() > 0:
            X_attacks = X[attack_mask]
            multiclass_pred[attack_mask] = self.multiclass_model.predict(X_attacks)

        return binary_pred, multiclass_pred

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test_binary: pd.Series,
        y_test_multiclass: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate both phases.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test_binary (pd.Series): Binary test labels.
            y_test_multiclass (pd.Series): Multi-class test labels.
        
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary containing
                performance metrics for both binary and multi-class phases.
        """
        # Binary evaluation
        binary_metrics = self.binary_model.evaluate(X_test, y_test_binary)

        # Multi-class evaluation on detected attacks
        binary_pred, multiclass_pred = self.predict(X_test)
        attack_mask = binary_pred == 0

        multiclass_metrics = {}
        if attack_mask.sum() > 0:
            y_test_attacks = y_test_multiclass[attack_mask]
            y_pred_attacks = multiclass_pred[attack_mask]

            multiclass_metrics = {
                "accuracy": accuracy_score(y_test_attacks, y_pred_attacks),
                "precision": precision_score(
                    y_test_attacks, y_pred_attacks, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test_attacks, y_pred_attacks, average="weighted"
                ),
                "f1_score": f1_score(
                    y_test_attacks, y_pred_attacks, average="weighted"
                ),
            }

        return {"binary": binary_metrics, "multiclass": multiclass_metrics}
