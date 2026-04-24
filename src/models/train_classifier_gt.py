from pathlib import Path
import json
import joblib
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
from sklearn.model_selection import train_test_split


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


def main():
    feature_paths = [
        "data/processed/features_gt/MOT17-02-DPM_features.csv",
        "data/processed/features_gt/MOT17-02-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-02-SDP_features.csv",
        "data/processed/features_gt/MOT17-04-DPM_features.csv",
        "data/processed/features_gt/MOT17-04-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-04-SDP_features.csv",
        "data/processed/features_gt/MOT17-05-DPM_features.csv",
        "data/processed/features_gt/MOT17-05-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-05-SDP_features.csv",
        "data/processed/features_gt/MOT17-09-DPM_features.csv",
        "data/processed/features_gt/MOT17-09-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-09-SDP_features.csv",
        "data/processed/features_gt/MOT17-10-DPM_features.csv",
        "data/processed/features_gt/MOT17-10-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-10-SDP_features.csv",
        "data/processed/features_gt/MOT17-11-DPM_features.csv",
        "data/processed/features_gt/MOT17-11-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-11-SDP_features.csv",
        "data/processed/features_gt/MOT17-13-DPM_features.csv",
        "data/processed/features_gt/MOT17-13-FRCNN_features.csv",
        "data/processed/features_gt/MOT17-13-SDP_features.csv",
    ]

    output_dir = Path("outputs/models/random_forest_gt")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_data(feature_paths)

    # Drop any rows with missing values
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "frame_diff_mean",
        "frame_diff_max",
        "flow_mean",
        "flow_max",
        "flow_std",
        "edge_diff_mean",
        "edge_density_prev",
        "edge_density_curr",
        "hist_bhattacharyya",
        "hist_corr",
    ]

    target_col = "label"

    X = df[feature_cols]
    y = df[target_col]

    # Keep label balance roughly consistent across split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # model = RandomForestClassifier(
    #     n_estimators=300,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     class_weight="balanced",
    #     random_state=42,
    #     n_jobs=-1,
    # )
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]
    y_prob = model.predict_proba(X_test)[:, 1]
    decision_threshold = 0.55
    y_pred = (y_prob >= decision_threshold).astype(int)

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
        "decision_threshold": decision_threshold,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    importances.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save test predictions with useful context
    results_df = df.loc[idx_test].copy()
    results_df = results_df.reset_index(drop=True)
    results_df["y_true"] = y_test.reset_index(drop=True)
    results_df["y_pred"] = y_pred
    results_df["y_prob_keep"] = y_prob
    results_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Save model
    joblib.dump(model, output_dir / "random_forest_gt_model.joblib")

    print("Training complete.")
    print(f"Saved model to: {output_dir / 'random_forest_gt_model.joblib'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    print(f"Saved feature importances to: {output_dir / 'feature_importance.csv'}")
    print(f"Saved test predictions to: {output_dir / 'test_predictions.csv'}")

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