import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

KNOWN_CLASSES = [
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "Brute Force",
    "Botnet",
    "Web Attack",
    "Infiltration",
]


def normalize_ground_truth(label):
    if pd.isna(label):
        return "UNKNOWN_LABEL"

    value = str(label).strip()

    if value.lower() == "benign":
        return "BENIGN"

    if value in [
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
        "Heartbleed",
    ]:
        return "DoS"

    if value == "DDoS":
        return "DDoS"

    if value == "PortScan":
        return "PortScan"

    if value in ["FTP-Patator", "SSH-Patator"]:
        return "Brute Force"

    if value == "Bot":
        return "Botnet"

    if value.startswith("Web Attack"):
        return "Web Attack"

    if value == "Infiltration":
        return "Infiltration"

    return "UNKNOWN_LABEL"


def evaluate_predictions(df, results_df):
    """
    Evaluate GTAE-IDS predictions against the original
    CIC-IDS2017 Label column.

    Labels are used only for evaluation after inference.
    """

    if "Label" not in df.columns:
        raise ValueError(
            "Ground-truth evaluation requires a 'Label' column."
        )

    if results_df.empty:
        raise ValueError(
            "No inference results available."
        )

    if len(df) != len(results_df):
        raise ValueError(
            f"Row count mismatch: data={len(df)}, "
            f"predictions={len(results_df)}"
        )

    y_true = [
        normalize_ground_truth(label)
        for label in df["Label"]
    ]

    y_pred = (
        results_df["detected_type"]
        .astype(str)
        .tolist()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=KNOWN_CLASSES,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=KNOWN_CLASSES,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=KNOWN_CLASSES,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=KNOWN_CLASSES,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=KNOWN_CLASSES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=KNOWN_CLASSES,
        columns=KNOWN_CLASSES,
    )

    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    report = classification_report(
        y_true,
        y_pred,
        labels=KNOWN_CLASSES,
        target_names=KNOWN_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Class"})
    )

    benign_total = sum(
        true == "BENIGN"
        for true in y_true
    )

    benign_false_positives = sum(
        true == "BENIGN" and pred != "BENIGN"
        for true, pred in zip(y_true, y_pred)
    )

    benign_fpr = (
        benign_false_positives / benign_total
        if benign_total
        else 0.0
    )

    evaluation_df = results_df.copy()

    evaluation_df.insert(
        1,
        "ground_truth",
        y_true,
    )

    evaluation_df["correct_prediction"] = (
        evaluation_df["ground_truth"]
        == evaluation_df["detected_type"]
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "benign_fpr": float(benign_fpr),
        "correct": int(
            evaluation_df["correct_prediction"].sum()
        ),
        "incorrect": int(
            (~evaluation_df["correct_prediction"]).sum()
        ),
        "total": len(evaluation_df),
        "confusion_matrix": cm_df,
        "classification_report": report_df,
        "evaluation_dataframe": evaluation_df,
    }
