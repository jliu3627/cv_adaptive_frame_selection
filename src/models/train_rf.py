from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


FEATURE_COLS = [
    "frame_diff_mean",
    "frame_diff_max",
    "frame_diff_p90",
    "frame_diff_p95",
    "frame_diff_p99",
    "changed_pixel_ratio_25",
    "changed_pixel_ratio_50",
    "flow_mean",
    "flow_max",
    "flow_std",
    "flow_p90",
    "flow_p95",
    "flow_p99",
    "flow_top10_mean",
    "edge_diff_mean",
    "edge_density_prev",
    "edge_density_curr",
    "hist_bhattacharyya",
    "hist_corr",
    "since_keep_frame_diff_mean",
    "since_keep_frame_diff_max",
    "since_keep_frame_diff_p90",
    "since_keep_frame_diff_p95",
    "since_keep_frame_diff_p99",
    "since_keep_changed_pixel_ratio_25",
    "since_keep_changed_pixel_ratio_50",
    "since_keep_flow_mean",
    "since_keep_flow_max",
    "since_keep_flow_std",
    "since_keep_flow_p90",
    "since_keep_flow_p95",
    "since_keep_flow_p99",
    "since_keep_flow_top10_mean",
    "since_keep_edge_diff_mean",
    "since_keep_edge_density_prev",
    "since_keep_edge_density_curr",
    "since_keep_hist_bhattacharyya",
    "since_keep_hist_corr",
    "frames_since_last_keep",
]

HIGH_MOTION_FEATURES = [
    "frame_diff_p95",
    "flow_p95",
    "flow_top10_mean",
    "since_keep_frame_diff_p95",
    "since_keep_flow_p95",
    "since_keep_flow_top10_mean",
]


def load_feature_data(feature_paths):
    dfs = []

    for path in feature_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        df = pd.read_csv(path)
        df["sequence"] = path.stem.replace("_features", "")
        dfs.append(df)

    if not dfs:
        raise ValueError("No feature files were loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def chronological_sequence_split(df, test_fraction=0.20):
    train_indices = []
    test_indices = []

    for _, seq_df in df.sort_values(["sequence", "frame"]).groupby("sequence"):
        split_idx = int(len(seq_df) * (1.0 - test_fraction))
        split_idx = min(max(split_idx, 1), len(seq_df) - 1)

        train_indices.extend(seq_df.index[:split_idx].tolist())
        test_indices.extend(seq_df.index[split_idx:].tolist())

    return train_indices, test_indices


def compute_high_motion_thresholds(train_df, quantile=0.80):
    thresholds = {}
    for feature in HIGH_MOTION_FEATURES:
        thresholds[feature] = float(train_df[feature].quantile(quantile))
    return thresholds


def apply_motion_augmented_labels(df, thresholds):
    df = df.copy()
    high_motion = pd.Series(False, index=df.index)

    for feature, threshold in thresholds.items():
        high_motion = high_motion | (df[feature] >= threshold)

    df["original_label"] = df["label"].astype(int)
    df["high_motion_label"] = high_motion.astype(int)
    df["training_label"] = (
        (df["original_label"] == 1) | (df["high_motion_label"] == 1)
    ).astype(int)
    return df


def save_sequence_diagnostics(all_predictions_df, test_predictions_df, output_dir: Path):
    accuracy_rows = []
    plot_dir = output_dir / "sequence_error_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sequence, seq_df in all_predictions_df.groupby("sequence"):
        test_seq_df = test_predictions_df[test_predictions_df["sequence"] == sequence]

        accuracy_rows.append(
            {
                "sequence": sequence,
                "num_frames": int(len(seq_df)),
                "num_test_frames": int(len(test_seq_df)),
                "full_accuracy": accuracy_score(seq_df["y_true"], seq_df["y_pred"]),
                "test_accuracy": (
                    accuracy_score(test_seq_df["y_true"], test_seq_df["y_pred"])
                    if len(test_seq_df) > 0
                    else None
                ),
                "false_skips": int(((seq_df["y_true"] == 1) & (seq_df["y_pred"] == 0)).sum()),
                "false_keeps": int(((seq_df["y_true"] == 0) & (seq_df["y_pred"] == 1)).sum()),
                "true_keep_rate": float(seq_df["y_true"].mean()),
                "pred_keep_rate": float(seq_df["y_pred"].mean()),
            }
        )

    sequence_metrics_df = pd.DataFrame(accuracy_rows).sort_values("sequence")
    sequence_metrics_df.to_csv(output_dir / "sequence_accuracy.csv", index=False)

    sequences = sequence_metrics_df["sequence"].tolist()
    for sequence in sequences:
        seq_df = all_predictions_df[all_predictions_df["sequence"] == sequence].sort_values("frame")
        seq_metrics = sequence_metrics_df[sequence_metrics_df["sequence"] == sequence].iloc[0]

        false_skip = (seq_df["y_true"] == 1) & (seq_df["y_pred"] == 0)
        false_keep = (seq_df["y_true"] == 0) & (seq_df["y_pred"] == 1)
        error_df = seq_df[false_skip | false_keep].copy()

        fig, ax = plt.subplots(figsize=(14, 3.5))

        ax.scatter(
            seq_df.loc[false_skip, "frame"],
            seq_df.loc[false_skip, "y_pred"],
            s=28,
            c="#d62728",
            alpha=0.90,
            label="False skip (true KEEP, pred SKIP)",
        )
        ax.scatter(
            seq_df.loc[false_keep, "frame"],
            seq_df.loc[false_keep, "y_pred"],
            s=28,
            c="#ff7f0e",
            alpha=0.85,
            label="False keep (true SKIP, pred KEEP)",
        )

        test_accuracy = seq_metrics["test_accuracy"]
        test_accuracy_text = "n/a" if pd.isna(test_accuracy) else f"{test_accuracy:.3f}"
        ax.set_title(
            f"{sequence} | full acc={seq_metrics['full_accuracy']:.3f} | "
            f"test acc={test_accuracy_text} | false skips={int(seq_metrics['false_skips'])}",
            fontsize=10,
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["SKIP", "KEEP"])
        ax.set_ylim(-0.25, 1.25)
        ax.set_xlim(seq_df["frame"].min(), seq_df["frame"].max())
        ax.set_xlabel("Frame")
        ax.set_ylabel("Predicted decision")
        ax.grid(axis="x", alpha=0.2)
        if error_df.empty:
            ax.text(
                0.5,
                0.5,
                "No false skips or false keeps",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.legend(loc="upper right")

        fig.tight_layout()
        safe_sequence = sequence.replace("/", "_")
        fig.savefig(plot_dir / f"{safe_sequence}_errors.png", dpi=200)
        plt.close(fig)


def main():
    feature_dir = Path("data/processed/features")
    model_output_dir = Path("outputs/models/random_forest")
    
    feature_paths = feature_dir.glob("*.csv")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_data(feature_paths)
    df = df.dropna().reset_index(drop=True)

    missing_features = [feature for feature in FEATURE_COLS if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "Feature CSVs are missing new feature columns. Rerun "
            f"src/features/extract_cv_features.py. Missing: {missing_features}"
        )

    idx_train, idx_test = chronological_sequence_split(df, test_fraction=0.20)
    high_motion_thresholds = compute_high_motion_thresholds(df.loc[idx_train], quantile=0.80)
    df = apply_motion_augmented_labels(df, high_motion_thresholds)

    feature_cols = FEATURE_COLS
    target_col = "training_label"

    X = df[feature_cols]
    y = df[target_col]

    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]
    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]

    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # NOTE: tune decision threshold here
    y_prob = model.predict_proba(X_test)[:, 1]
    decision_threshold = 0.50
    y_pred = (y_prob >= decision_threshold).astype(int)
    y_prob_all = model.predict_proba(X)[:, 1]
    y_pred_all = (y_prob_all >= decision_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "num_train_samples": int(len(X_train)),
        "num_test_samples": int(len(X_test)),
        "label_distribution_full": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "label_distribution_train": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
        "label_distribution_test": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
        "original_label_distribution_full": {
            str(k): int(v) for k, v in df["original_label"].value_counts().to_dict().items()
        },
        "high_motion_label_distribution_full": {
            str(k): int(v) for k, v in df["high_motion_label"].value_counts().to_dict().items()
        },
        "split_strategy": "last_20_percent_per_sequence_by_frame",
        "decision_threshold": decision_threshold,
    }

    with open(model_output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(model_output_dir / "high_motion_thresholds.json", "w") as f:
        json.dump(high_motion_thresholds, f, indent=4)

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    importances.to_csv(model_output_dir / "feature_importance.csv", index=False)

    # Save test predictions with useful context
    results_df = df.loc[idx_test].copy()
    results_df = results_df.reset_index(drop=True)
    results_df["y_true"] = y_test.reset_index(drop=True)
    results_df["y_pred"] = y_pred
    results_df["y_prob_keep"] = y_prob
    results_df.to_csv(model_output_dir / "test_predictions.csv", index=False)

    all_predictions_df = df.copy()
    all_predictions_df["y_true"] = y.reset_index(drop=True)
    all_predictions_df["y_pred"] = y_pred_all
    all_predictions_df["y_prob_keep"] = y_prob_all
    all_predictions_df.to_csv(model_output_dir / "all_predictions.csv", index=False)
    save_sequence_diagnostics(all_predictions_df, results_df, model_output_dir)

    # Save model
    joblib.dump(model, model_output_dir / "random_forest_gt_model.joblib")

    print("Training complete.")
    print(f"Saved model to: {model_output_dir / 'random_forest_gt_model.joblib'}")
    print(f"Saved metrics to: {model_output_dir / 'metrics.json'}")
    print(f"Saved high-motion thresholds to: {model_output_dir / 'high_motion_thresholds.json'}")
    print(f"Saved feature importances to: {model_output_dir / 'feature_importance.csv'}")
    print(f"Saved test predictions to: {model_output_dir / 'test_predictions.csv'}")
    print(f"Saved all predictions to: {model_output_dir / 'all_predictions.csv'}")
    print(f"Saved sequence accuracy to: {model_output_dir / 'sequence_accuracy.csv'}")
    print(f"Saved sequence error plots to: {model_output_dir / 'sequence_error_plots'}")

    print("\n=== Test Metrics ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Feature Importances ===")
    print(importances)


if __name__ == "__main__":
    main()
