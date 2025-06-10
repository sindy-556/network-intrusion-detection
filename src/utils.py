"""Utility functions for the intrusion detection system."""

import json
import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            Defaults to "config.json".
    
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from the file,
            or empty dictionary if file doesn't exist.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    """Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        config_path (str, optional): Path where the configuration should be saved.
            Defaults to "config.json".
    """
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def format_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
    """Format metrics dictionary for display.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metric names and values.
    
    Returns:
        pd.DataFrame: Formatted dataframe with metric names and rounded values.
    """
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df["Value"] = df["Value"].round(4)
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe to summarise.
    
    Returns:
        Dict[str, Any]: Dictionary containing dataset summary information
            including shape, columns, data types, missing values, duplicates,
            and target distribution if labels column exists.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
    }

    # Add target distribution if labels column exists
    if "labels" in df.columns:
        summary["target_distribution"] = df["labels"].value_counts().to_dict()

    return summary


def validate_data(df: pd.DataFrame, expected_columns: List[str]) -> tuple[bool, str]:
    """Validate uploaded data format.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        expected_columns (List[str]): List of required column names.
    
    Returns:
        tuple[bool, str]: Tuple containing validation status (True if valid)
            and descriptive message about the validation result.
    """
    # Check if dataframe is empty
    if df.empty:
        return False, "Uploaded file is empty"

    # Check for required columns
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"

    # Check for invalid values in specific columns
    if "su_attempted" in df.columns:
        invalid_su = df["su_attempted"].isin([2]).sum()
        if invalid_su > 0:
            return False, f"Found {invalid_su} invalid values in 'su_attempted' column"

    return True, "Data validation passed"


def display_info_box(title: str, content: str, box_type: str = "info"):
    """Display an information box in Streamlit.
    
    Args:
        title (str): Title text for the information box.
        content (str): Content text to display.
        box_type (str, optional): Type of box to display. Options are
            'info', 'success', 'warning', or 'error'. Defaults to "info".
    """
    if box_type == "info":
        st.info(f"**{title}**: {content}")
    elif box_type == "success":
        st.success(f"**{title}**: {content}")
    elif box_type == "warning":
        st.warning(f"**{title}**: {content}")
    elif box_type == "error":
        st.error(f"**{title}**: {content}")
