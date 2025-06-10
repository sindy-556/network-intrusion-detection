"""Data processing module for NSL-KDD dataset."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    """Handles data loading, preprocessing, and splitting.
    
    A comprehensive data processor for NSL-KDD dataset that provides functionality
    for loading, preprocessing, and preparing network intrusion detection data.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.attack_mapping = self._get_attack_mapping()

    @staticmethod
    def _get_attack_mapping() -> Dict[str, str]:
        """Returns mapping of specific attacks to attack categories.
        
        Returns:
            Dict[str, str]: Mapping from specific attack names to their general
                categories (DoS, Probe, R2L, U2R, normal).
        """
        return {
            "normal": "normal",
            # DoS attacks
            "back": "DoS",
            "land": "DoS",
            "neptune": "DoS",
            "pod": "DoS",
            "smurf": "DoS",
            "teardrop": "DoS",
            "mailbomb": "DoS",
            "apache2": "DoS",
            "processtable": "DoS",
            "udpstorm": "DoS",
            # Probe attacks
            "ipsweep": "Probe",
            "nmap": "Probe",
            "portsweep": "Probe",
            "satan": "Probe",
            "mscan": "Probe",
            "saint": "Probe",
            # R2L attacks
            "ftp_write": "R2L",
            "guess_passwd": "R2L",
            "imap": "R2L",
            "multihop": "R2L",
            "phf": "R2L",
            "spy": "R2L",
            "warezclient": "R2L",
            "warezmaster": "R2L",
            "sendmail": "R2L",
            "named": "R2L",
            "snmpgetattack": "R2L",
            "snmpguess": "R2L",
            "xlock": "R2L",
            "xsnoop": "R2L",
            "worm": "R2L",
            # U2R attacks
            "buffer_overflow": "U2R",
            "loadmodule": "U2R",
            "perl": "U2R",
            "rootkit": "U2R",
            "httptunnel": "U2R",
            "ps": "U2R",
            "sqlattack": "U2R",
            "xterm": "U2R",
        }

    def load_data(
        self, filepath: str, column_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load NSL-KDD dataset from file.
        
        Args:
            filepath (str): Path to the CSV file containing the dataset.
            column_names (Optional[List[str]]): List of column names to use.
                If None, default NSL-KDD column names are used.
        
        Returns:
            pd.DataFrame: Loaded dataset with proper column names and cleaned data.
        """
        if column_names is None:
            column_names = self._get_default_column_names()

        # Try to read the file
        try:
            # First, try reading with headers
            df = pd.read_csv(filepath)

            # Check if it has the expected number of columns
            if len(df.columns) == len(column_names):
                # If columns match but have different names, rename them
                df.columns = column_names
            elif len(df.columns) == len(column_names) - 1:
                # KDDTest+ and KDDTrain+ might not have the 'difficulty' column
                df.columns = column_names[:-1]
            else:
                # Try reading without headers
                df = pd.read_csv(filepath, names=column_names, header=None)
        except Exception:
            # If all else fails, read without headers
            df = pd.read_csv(filepath, names=column_names, header=None)

        # Clean su_attempted column (remove invalid values)
        if "su_attempted" in df.columns:
            df = df[df["su_attempted"] != 2].copy()

        return df

    def _get_default_column_names(self) -> List[str]:
        """Get default column names for NSL-KDD dataset.
        
        Returns:
            List[str]: List of 42 standard column names used in the NSL-KDD dataset.
        """
        return [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "labels",
            "difficulty",
        ]

    def create_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary and multi-class labels.
        
        Creates both binary classification labels (normal vs attack) and
        multi-class labels (normal, DoS, Probe, R2L, U2R) from the original
        attack labels.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'labels' column containing
                original attack names.
        
        Returns:
            pd.DataFrame: Dataframe with additional 'labels2' (binary) and
                'labels5' (multi-class) columns containing encoded labels.
        """
        df = df.copy()

        # Multi-class labels (5 classes)
        df["labels5"] = df["labels"].map(self.attack_mapping)

        # Binary labels
        df["labels2"] = df["labels"].apply(
            lambda x: "normal" if x == "normal" else "attack"
        )

        # Encode labels
        for label_col in ["labels2", "labels5"]:
            le = LabelEncoder()
            df[label_col] = le.fit_transform(df[label_col])
            self.label_encoders[label_col] = le

        return df

    def get_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify different types of features.
        
        Categorises features into binary, categorical, and continuous types
        for appropriate preprocessing.
        
        Args:
            df (pd.DataFrame): Input dataframe to analyse.
        
        Returns:
            Dict[str, List[str]]: Dictionary with keys 'binary', 'categorical',
                and 'continuous' mapping to lists of column names.
        """
        # Binary features
        binary_cols = [
            col
            for col in df.columns
            if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1})
        ]

        # Categorical features
        categorical_cols = [col for col in df.columns if df[col].dtype == "object"]

        # Continuous features
        continuous_cols = [
            col
            for col in df.columns
            if col
            not in binary_cols + categorical_cols + ["labels", "labels2", "labels5"]
            and (df[col].dtype == "int64" or df[col].dtype == "float64")
        ]

        return {
            "binary": binary_cols,
            "categorical": categorical_cols,
            "continuous": continuous_cols,
        }

    def prepare_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Tuple:
        """Prepare features and split data.
        
        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.DataFrame): Target dataframe.
            test_size (float, optional): Proportion of dataset to include in the
                test split. Defaults to 0.3.
            random_state (int, optional): Random state for reproducible splits.
                Defaults to 42.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test split datasets.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def get_label_names(self, label_type: str = "labels5") -> List[str]:
        """Get label names for a specific label encoder.
        
        Args:
            label_type (str, optional): Type of labels to retrieve names for.
                Either 'labels2' for binary or 'labels5' for multi-class.
                Defaults to "labels5".
        
        Returns:
            List[str]: List of class names in the order of their encoded values.
        """
        if label_type in self.label_encoders:
            return list(self.label_encoders[label_type].classes_)
        return []
