"""
Visualization module for Explanations and Risk Distribution.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_importance(
    importances: List[Tuple[str, float]], 
    title: str = "Feature Importance",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots horizontal bar chart for feature importances.
    """
    if not importances:
        return
        
    features = [x[0] for x in importances]
    scores = [x[1] for x in importances]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=features, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_risk_distribution(
    results: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots the distribution of risk scores categorized by severity.
    """
    if not results:
        return
        
    scores = [r["risk_score"] for r in results]
    severities = [r["severity"] for r in results]
    
    df = pd.DataFrame({"Risk Score": scores, "Severity": severities})
    
    # Sort order for severity
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    palette = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "orange", "CRITICAL": "red"}
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x="Risk Score", 
        hue="Severity", 
        hue_order=[o for o in order if o in df["Severity"].unique()],
        multiple="stack",
        palette=palette,
        bins=20
    )
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score (0-100)")
    plt.ylabel("Count")
    plt.xlim(0, 100)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
