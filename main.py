"""
This performs the complete project in one command for one entire run.

    DPM + FRCNN + SDP

Pipeline order:
    1. Read MOT17 GT annotations
    2. Generate GT-based keep/skip labels
    3. Extract classical CV features
    4. Train Random Forest adaptive frame selector
    5. Run full YOLO baseline
    6. Run RF adaptive YOLO inference
    7. Run every-k-frame YOLO baseline
    8. Evaluate adaptive reduced detection vs full YOLO
    9. Evaluate k-frame baseline vs full YOLO
    10. Evaluate adaptive detections vs MOT17 GT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split



# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

SRC_SUBDIRS = [
    SRC_DIR / "data",
    SRC_DIR / "labeling",
    SRC_DIR / "features",
    SRC_DIR / "models",
    SRC_DIR / "inference",
    SRC_DIR / "evaluation",
]

for folder in SRC_SUBDIRS:
    sys.path.insert(0, str(folder))



# Imports from your existing project files

try:
    from read_mot17 import save_gt_for_sequence
    from generate_labels import generate_labels_for_sequence_from_gt
    from extract_cv_features import extract_features_for_sequence

    from run_full_yolo_baseline import run_full_yolo_for_sequence
    from run_rf_adaptive_model_yolo import run_adaptive_inference_for_sequence
    from run_kframe_yolo import run_kframe_yolo_for_sequence

    from evaluate_reduced_detection import (
        aggregate_results as aggregate_adaptive_results,
        evaluate_sequence as evaluate_adaptive_reduced_sequence,
        save_json as save_eval_json,
    )

    from evaluate_kframe_detection import (
        aggregate_results as aggregate_kframe_results,
        evaluate_sequence as evaluate_kframe_sequence,
    )

    from evaluate_adaptive_vs_gt import (
        evaluate_sequence as evaluate_adaptive_vs_gt_sequence,
    )

except ImportError as error:
    raise ImportError(
        "\nCould not import one or more project modules.\n"
        "Make sure main.py is placed in the root folder:\n\n"
        "    cv_adaptive_frame_selection-main/main.py\n\n"
        "And make sure your src folder contains:\n"
        "    src/data/read_mot17.py\n"
        "    src/labeling/generate_labels.py\n"
        "    src/features/extract_cv_features.py\n"
        "    src/inference/run_full_yolo_baseline.py\n"
        "    src/inference/run_rf_adaptive_model_yolo.py\n"
        "    src/inference/run_kframe_yolo.py\n"
        "    src/evaluation/evaluate_reduced_detection.py\n"
        "    src/evaluation/evaluate_kframe_detection.py\n"
        "    src/evaluation/evaluate_adaptive_vs_gt.py\n"
    ) from error


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None



# Constants

DEFAULT_SEQUENCE_IDS = ["02", "04", "05", "09", "10", "11", "13"]
DEFAULT_VARIANTS = ["DPM", "FRCNN", "SDP"]

FEATURE_COLS = [
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



# Utility functions

def print_step(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def ensure_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def build_sequence_names(
    sequence_ids: Iterable[str],
    variants: Iterable[str],
) -> List[str]:
    sequence_names = []

    for seq_id in sequence_ids:
        seq_id = str(seq_id).zfill(2)

        for variant in variants:
            variant = variant.upper()
            sequence_names.append(f"MOT17-{seq_id}-{variant}")

    return sequence_names


def discover_sequences(
    base_dir: Path,
    variants: Iterable[str],
) -> List[str]:
    """
    Discover available MOT17 sequences from disk.

    Example detected folders:
        MOT17-02-DPM
        MOT17-02-FRCNN
        MOT17-02-SDP
    """
    if not base_dir.exists():
        return []

    variants = {v.upper() for v in variants}

    found = []

    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue

        name = path.name

        for variant in variants:
            if name.endswith(f"-{variant}"):
                found.append(name)
                break

    return found


def validate_train_sequence(sequence_dir: Path) -> None:
    ensure_exists(sequence_dir, "MOT17 sequence folder")
    ensure_exists(sequence_dir / "img1", "MOT17 img1 image folder")
    ensure_exists(sequence_dir / "gt" / "gt.txt", "MOT17 ground-truth file")


def validate_test_sequence(sequence_dir: Path) -> None:
    ensure_exists(sequence_dir, "MOT17 test sequence folder")
    ensure_exists(sequence_dir / "img1", "MOT17 test img1 image folder")


def load_feature_data(feature_paths: Iterable[Path]) -> pd.DataFrame:
    frames = []

    for path in sorted(feature_paths):
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")

        df = pd.read_csv(path)
        df["sequence"] = path.stem.replace("_features", "")
        frames.append(df)

    if not frames:
        raise ValueError("No feature files were loaded.")

    return pd.concat(frames, ignore_index=True)


def check_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLS if col not in df.columns]

    if missing:
        raise ValueError(
            "Feature CSV files are missing required columns:\n"
            + "\n".join(missing)
        )

    if "label" not in df.columns:
        raise ValueError("Feature CSV files must contain a 'label' column.")



# Stage 1: Read MOT17 GT annotations

def stage_1_read_gt(train_base_dir: Path, train_sequences: List[str]) -> None:
    print_step("STEP 1: Reading MOT17 ground-truth annotations")

    output_dir = PROJECT_ROOT / "data/interim/gt_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in train_sequences:
        sequence_dir = train_base_dir / seq_name
        validate_train_sequence(sequence_dir)

        output_csv = output_dir / f"{seq_name}_gt.csv"

        print(f"\nReading GT for {seq_name}")
        save_gt_for_sequence(
            sequence_dir=sequence_dir,
            output_csv=output_csv,
        )



# Stage 2: Generate GT labels

def stage_2_generate_labels(
    train_sequences: List[str],
    motion_threshold: float,
) -> None:
    print_step("STEP 2: Generating adaptive keep/skip labels from GT")

    gt_dir = PROJECT_ROOT / "data/interim/gt_annotations"
    output_dir = PROJECT_ROOT / "data/interim/gt_label"
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in train_sequences:
        gt_csv = gt_dir / f"{seq_name}_gt.csv"
        output_csv = output_dir / f"{seq_name}_labels.csv"

        print(f"\nGenerating labels for {seq_name}")

        generate_labels_for_sequence_from_gt(
            gt_csv=gt_csv,
            output_csv=output_csv,
            motion_threshold=motion_threshold,
        )



# Stage 3: Extract CV features

def stage_3_extract_features(
    train_base_dir: Path,
    train_sequences: List[str],
) -> None:
    print_step("STEP 3: Extracting classical CV features")

    labels_dir = PROJECT_ROOT / "data/interim/gt_label"
    output_dir = PROJECT_ROOT / "data/processed/features"
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in train_sequences:
        sequence_dir = train_base_dir / seq_name
        labels_csv = labels_dir / f"{seq_name}_labels.csv"
        output_csv = output_dir / f"{seq_name}_features.csv"

        print(f"\nExtracting features for {seq_name}")

        extract_features_for_sequence(
            sequence_dir=sequence_dir,
            labels_csv=labels_csv,
            output_csv=output_csv,
        )



# Stage 4: Train Random Forest

def stage_4_train_random_forest(
    train_sequences: List[str],
    decision_threshold: float,
) -> Path:
    print_step("STEP 4: Training Random Forest adaptive frame-selection model")

    feature_dir = PROJECT_ROOT / "data/processed/features"
    output_dir = PROJECT_ROOT / "outputs/models/random_forest_all_variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = [
        feature_dir / f"{seq_name}_features.csv"
        for seq_name in train_sequences
    ]

    df = load_feature_data(feature_paths)
    df = df.dropna().reset_index(drop=True)

    check_feature_columns(df)

    X = df[FEATURE_COLS]
    y = df["label"].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The labels contain only one class. "
            "Try changing --motion-threshold or check label generation."
        )

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

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

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
        "num_total_samples": int(len(df)),
        "num_train_samples": int(len(X_train)),
        "num_test_samples": int(len(X_test)),
        "decision_threshold": float(decision_threshold),
        "trained_sequences": train_sequences,
        "label_distribution": {
            str(k): int(v)
            for k, v in y.value_counts().sort_index().to_dict().items()
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    importance_path = output_dir / "feature_importance.csv"
    importances.to_csv(importance_path, index=False)

    predictions_df = df.loc[idx_test].copy().reset_index(drop=True)
    predictions_df["y_true"] = y_test.reset_index(drop=True)
    predictions_df["y_pred"] = y_pred
    predictions_df["y_prob_keep"] = y_prob

    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    model_path = output_dir / "random_forest_gt_model.joblib"
    joblib.dump(model, model_path)

    print("\nRandom Forest model saved:")
    print(model_path)

    print("\nMetrics saved:")
    print(metrics_path)

    print("\nTest metrics:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

    print("\nFeature importances:")
    print(importances)

    return model_path



# Stage 5: Full YOLO baseline

def stage_5_full_yolo_baseline(
    base_dir: Path,
    sequences: List[str],
    split_name: str,
    yolo_model,
    yolo_conf: float,
) -> None:
    print_step(f"STEP 5: Running full YOLO baseline on {split_name} sequences")

    if split_name == "train":
        output_dir = PROJECT_ROOT / "outputs/full_yolo_baseline_gt/json"
    else:
        output_dir = PROJECT_ROOT / "outputs/test/full_yolo_baseline/json"

    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name

        if split_name == "train":
            validate_train_sequence(sequence_dir)
        else:
            validate_test_sequence(sequence_dir)

        output_json = output_dir / f"{seq_name}_full_yolo_detections.json"

        print(f"\nRunning full YOLO for {seq_name}")

        run_full_yolo_for_sequence(
            sequence_dir=sequence_dir,
            yolo_model=yolo_model,
            output_json=output_json,
            yolo_conf_thresh=yolo_conf,
        )



# Stage 6: Adaptive RF YOLO inference

def stage_6_adaptive_inference(
    base_dir: Path,
    sequences: List[str],
    split_name: str,
    model_path: Path,
    yolo_model,
    keep_threshold: float,
    yolo_conf: float,
    max_skip_streak: int,
) -> None:
    print_step(f"STEP 6: Running RF adaptive YOLO inference on {split_name} sequences")

    rf_model = joblib.load(model_path)

    if split_name == "train":
        output_root = PROJECT_ROOT / "outputs/adaptive_inference_gt"
    else:
        output_root = PROJECT_ROOT / "outputs/test/adaptive_inference"

    decisions_dir = output_root / "decisions"
    detections_dir = output_root / "detections"
    reduced_json_dir = output_root / "reduced_yolo_json"
    kept_frames_dir = output_root / "kept_frames"

    decisions_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)
    reduced_json_dir.mkdir(parents=True, exist_ok=True)
    kept_frames_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name

        if split_name == "train":
            validate_train_sequence(sequence_dir)
        else:
            validate_test_sequence(sequence_dir)

        print(f"\nRunning adaptive inference for {seq_name}")

        run_adaptive_inference_for_sequence(
            sequence_dir=sequence_dir,
            rf_model=rf_model,
            yolo_model=yolo_model,
            decisions_output_csv=decisions_dir / f"{seq_name}_decisions.csv",
            detections_output_csv=detections_dir / f"{seq_name}_adaptive_detections.csv",
            reduced_json_output=reduced_json_dir / f"{seq_name}_reduced_yolo_detections.json",
            kept_frames_output=kept_frames_dir / f"{seq_name}_kept_frames.json",
            keep_threshold=keep_threshold,
            yolo_conf_thresh=yolo_conf,
            max_skip_streak=max_skip_streak,
        )



# Stage 7: Every-k-frame YOLO baseline

def stage_7_kframe_baseline(
    base_dir: Path,
    sequences: List[str],
    split_name: str,
    yolo_model,
    k: int,
    yolo_conf: float,
) -> None:
    print_step(f"STEP 7: Running every-{k}-frame YOLO baseline on {split_name} sequences")

    if split_name == "train":
        output_root = PROJECT_ROOT / "outputs/kframe_yolo_gt"
    else:
        output_root = PROJECT_ROOT / "outputs/test/kframe_yolo"

    detections_dir = output_root / "reduced_yolo_json"
    kept_frames_dir = output_root / "kept_frames"

    detections_dir.mkdir(parents=True, exist_ok=True)
    kept_frames_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name

        if split_name == "train":
            validate_train_sequence(sequence_dir)
        else:
            validate_test_sequence(sequence_dir)

        print(f"\nRunning every-{k}-frame YOLO for {seq_name}")

        run_kframe_yolo_for_sequence(
            sequence_dir=sequence_dir,
            yolo_model=yolo_model,
            output_json=detections_dir / f"{seq_name}_k{k}_reduced_yolo_detections.json",
            kept_frames_output=kept_frames_dir / f"{seq_name}_k{k}_kept_frames.json",
            k=k,
            yolo_conf_thresh=yolo_conf,
            always_keep_first=True,
        )



# Stage 8: Evaluate adaptive reduced detections vs full YOLO


def stage_8_evaluate_adaptive_reduced(
    sequences: List[str],
    split_name: str,
    iou_thresh: float,
    max_temporal_gap: int,
    match_class: bool,
) -> None:
    print_step(
        f"STEP 8: Evaluating adaptive reduced detections against full YOLO on {split_name}"
    )

    if split_name == "train":
        full_yolo_dir = PROJECT_ROOT / "outputs/full_yolo_baseline_gt/json"
        reduced_yolo_dir = PROJECT_ROOT / "outputs/adaptive_inference_gt/reduced_yolo_json"
        kept_frames_dir = PROJECT_ROOT / "outputs/adaptive_inference_gt/kept_frames"
        output_dir = PROJECT_ROOT / "outputs/adaptive_inference_gt/reduced_detection_eval"
        summary_name = "all_sequences_reduced_detection_eval.json"
    else:
        full_yolo_dir = PROJECT_ROOT / "outputs/test/full_yolo_baseline/json"
        reduced_yolo_dir = PROJECT_ROOT / "outputs/test/adaptive_inference/reduced_yolo_json"
        kept_frames_dir = PROJECT_ROOT / "outputs/test/adaptive_inference/kept_frames"
        output_dir = PROJECT_ROOT / "outputs/test/adaptive_inference/reduced_detection_eval"
        summary_name = "all_sequences_test_reduced_detection_eval.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seq_name in sequences:
        print(f"\nEvaluating adaptive reduced detections for {seq_name}")

        result = evaluate_adaptive_reduced_sequence(
            seq_name=seq_name,
            full_yolo_dir=full_yolo_dir,
            reduced_yolo_dir=reduced_yolo_dir,
            kept_frames_dir=kept_frames_dir,
            output_dir=output_dir,
            iou_thresh=iou_thresh,
            max_temporal_gap=max_temporal_gap,
            match_class=match_class,
        )

        results.append(result)

    aggregate = aggregate_adaptive_results(results)

    save_eval_json(
        {
            "per_sequence": results,
            "aggregate": aggregate,
        },
        output_dir / summary_name,
    )

    print("\nAdaptive reduced detection aggregate:")
    print(json.dumps(aggregate, indent=2))



# Stage 9: Evaluate every-k-frame detections vs full YOLO

def stage_9_evaluate_kframe(
    sequences: List[str],
    split_name: str,
    k: int,
    iou_thresh: float,
    max_temporal_gap: int,
    match_class: bool,
) -> None:
    print_step(
        f"STEP 9: Evaluating every-{k}-frame baseline against full YOLO on {split_name}"
    )

    if split_name == "train":
        full_yolo_dir = PROJECT_ROOT / "outputs/full_yolo_baseline_gt/json"
        kframe_yolo_dir = PROJECT_ROOT / "outputs/kframe_yolo_gt/reduced_yolo_json"
        kept_frames_dir = PROJECT_ROOT / "outputs/kframe_yolo_gt/kept_frames"
        output_dir = PROJECT_ROOT / "outputs/kframe_yolo_gt/evaluation"
        summary_name = f"all_sequences_k{k}_detection_eval.json"
    else:
        full_yolo_dir = PROJECT_ROOT / "outputs/test/full_yolo_baseline/json"
        kframe_yolo_dir = PROJECT_ROOT / "outputs/test/kframe_yolo/reduced_yolo_json"
        kept_frames_dir = PROJECT_ROOT / "outputs/test/kframe_yolo/kept_frames"
        output_dir = PROJECT_ROOT / "outputs/test/kframe_yolo/evaluation"
        summary_name = f"all_sequences_test_k{k}_detection_eval.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seq_name in sequences:
        print(f"\nEvaluating every-{k}-frame baseline for {seq_name}")

        result = evaluate_kframe_sequence(
            seq_name=seq_name,
            k=k,
            full_yolo_dir=full_yolo_dir,
            kframe_yolo_dir=kframe_yolo_dir,
            kept_frames_dir=kept_frames_dir,
            output_dir=output_dir,
            iou_thresh=iou_thresh,
            max_temporal_gap=max_temporal_gap,
            match_class=match_class,
        )

        results.append(result)

    aggregate = aggregate_kframe_results(results)

    save_eval_json(
        {
            "per_sequence": results,
            "aggregate": aggregate,
        },
        output_dir / summary_name,
    )

    print("\nK-frame detection aggregate:")
    print(json.dumps(aggregate, indent=2))



# Stage 10: Evaluate adaptive detections directly against MOT17 GT

def stage_10_evaluate_adaptive_vs_gt(
    train_sequences: List[str],
    iou_thresh: float,
) -> None:
    print_step("STEP 10: Evaluating adaptive detections directly against MOT17 GT")

    gt_dir = PROJECT_ROOT / "data/interim/gt_annotations"
    pred_dir = PROJECT_ROOT / "outputs/adaptive_inference_gt/detections"
    decisions_dir = PROJECT_ROOT / "outputs/adaptive_inference_gt/decisions"
    output_dir = PROJECT_ROOT / "outputs/evaluation/adaptive_vs_gt_all_variants"

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for seq_name in train_sequences:
        print(f"\nEvaluating adaptive-vs-GT for {seq_name}")

        metrics, per_frame_df = evaluate_adaptive_vs_gt_sequence(
            gt_csv=gt_dir / f"{seq_name}_gt.csv",
            pred_csv=pred_dir / f"{seq_name}_adaptive_detections.csv",
            decisions_csv=decisions_dir / f"{seq_name}_decisions.csv",
            iou_thresh=iou_thresh,
        )

        metrics["sequence"] = seq_name
        summary.append(metrics)

        with open(output_dir / f"{seq_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        per_frame_df.to_csv(
            output_dir / f"{seq_name}_per_frame.csv",
            index=False,
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAdaptive-vs-GT summary saved:")
    print(output_dir / "summary.json")


# Argument parser

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete MOT17 adaptive frame-selection project."
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=["DPM", "FRCNN", "SDP"],
        help="MOT17 detector variants to include.",
    )

    parser.add_argument(
        "--sequence-ids",
        nargs="+",
        default=DEFAULT_SEQUENCE_IDS,
        help="MOT17 train sequence IDs.",
    )

    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=20.0,
        help="Motion threshold used during GT label generation.",
    )

    parser.add_argument(
        "--rf-decision-threshold",
        type=float,
        default=0.55,
        help="Threshold for reporting RF test metrics.",
    )

    parser.add_argument(
        "--adaptive-keep-threshold",
        type=float,
        default=0.45,
        help="RF keep-probability threshold used during adaptive inference.",
    )

    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--max-skip-streak",
        type=int,
        default=5,
        help="Maximum number of consecutive skipped frames before forced YOLO refresh.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Every-k-frame baseline value.",
    )

    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for detection evaluation.",
    )

    parser.add_argument(
        "--max-temporal-gap",
        type=int,
        default=5,
        help="Maximum temporal gap for reduced detection evaluation.",
    )

    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Run only GT reading, label generation, feature extraction, and RF training.",
    )

    parser.add_argument(
        "--skip-kframe",
        action="store_true",
        help="Skip every-k-frame baseline and its evaluation.",
    )

    parser.add_argument(
        "--skip-gt-eval",
        action="store_true",
        help="Skip adaptive-vs-GT evaluation.",
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip MOT17 test split if present.",
    )

    return parser.parse_args()


# Main function

def main() -> None:
    args = parse_args()

    train_base_dir = PROJECT_ROOT / "data/raw/MOT17/train"
    test_base_dir = PROJECT_ROOT / "data/raw/MOT17/test"

    ensure_exists(train_base_dir, "MOT17 train folder")

    variants = [v.upper() for v in args.variants]

    expected_train_sequences = build_sequence_names(
        sequence_ids=args.sequence_ids,
        variants=variants,
    )

    existing_expected_train_sequences = [
        seq_name
        for seq_name in expected_train_sequences
        if (train_base_dir / seq_name).exists()
    ]

    if existing_expected_train_sequences:
        train_sequences = existing_expected_train_sequences
    else:
        train_sequences = discover_sequences(
            base_dir=train_base_dir,
            variants=variants,
        )

    if not train_sequences:
        raise FileNotFoundError(
            "No MOT17 train sequences found.\n\n"
            "Expected folders like:\n"
            "    data/raw/MOT17/train/MOT17-02-DPM\n"
            "    data/raw/MOT17/train/MOT17-02-FRCNN\n"
            "    data/raw/MOT17/train/MOT17-02-SDP\n"
        )

    test_sequences = discover_sequences(
        base_dir=test_base_dir,
        variants=variants,
    )

    print("\nSelected train sequences:")
    for seq_name in train_sequences:
        print(f"  - {seq_name}")

    if test_sequences:
        print("\nSelected test sequences:")
        for seq_name in test_sequences:
            print(f"  - {seq_name}")
    else:
        print("\nNo MOT17 test sequences found. Test split will be skipped.")

    # Core training pipeline
    stage_1_read_gt(
        train_base_dir=train_base_dir,
        train_sequences=train_sequences,
    )

    stage_2_generate_labels(
        train_sequences=train_sequences,
        motion_threshold=args.motion_threshold,
    )

    stage_3_extract_features(
        train_base_dir=train_base_dir,
        train_sequences=train_sequences,
    )

    model_path = stage_4_train_random_forest(
        train_sequences=train_sequences,
        decision_threshold=args.rf_decision_threshold,
    )

    if args.skip_yolo:
        print("\nDONE: Preprocessing and Random Forest training completed.")
        print("YOLO inference and evaluation were skipped because --skip-yolo was used.")
        return

    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed.\n"
            "Install it using:\n\n"
            "    pip install ultralytics\n\n"
            "or:\n\n"
            "    pip install -r requirements.txt\n"
        )

    print_step("Loading YOLO model")
    yolo_model = YOLO("yolov8n.pt")

    # Train split YOLO and evaluation
    stage_5_full_yolo_baseline(
        base_dir=train_base_dir,
        sequences=train_sequences,
        split_name="train",
        yolo_model=yolo_model,
        yolo_conf=args.yolo_conf,
    )

    stage_6_adaptive_inference(
        base_dir=train_base_dir,
        sequences=train_sequences,
        split_name="train",
        model_path=model_path,
        yolo_model=yolo_model,
        keep_threshold=args.adaptive_keep_threshold,
        yolo_conf=args.yolo_conf,
        max_skip_streak=args.max_skip_streak,
    )

    if not args.skip_kframe:
        stage_7_kframe_baseline(
            base_dir=train_base_dir,
            sequences=train_sequences,
            split_name="train",
            yolo_model=yolo_model,
            k=args.k,
            yolo_conf=args.yolo_conf,
        )

    stage_8_evaluate_adaptive_reduced(
        sequences=train_sequences,
        split_name="train",
        iou_thresh=args.iou_thresh,
        max_temporal_gap=args.max_temporal_gap,
        match_class=True,
    )

    if not args.skip_kframe:
        stage_9_evaluate_kframe(
            sequences=train_sequences,
            split_name="train",
            k=args.k,
            iou_thresh=args.iou_thresh,
            max_temporal_gap=args.max_temporal_gap,
            match_class=True,
        )

    if not args.skip_gt_eval:
        stage_10_evaluate_adaptive_vs_gt(
            train_sequences=train_sequences,
            iou_thresh=args.iou_thresh,
        )

    # Optional MOT17 test split inference/evaluation
    if test_sequences and not args.skip_test:
        stage_5_full_yolo_baseline(
            base_dir=test_base_dir,
            sequences=test_sequences,
            split_name="test",
            yolo_model=yolo_model,
            yolo_conf=args.yolo_conf,
        )

        stage_6_adaptive_inference(
            base_dir=test_base_dir,
            sequences=test_sequences,
            split_name="test",
            model_path=model_path,
            yolo_model=yolo_model,
            keep_threshold=args.adaptive_keep_threshold,
            yolo_conf=args.yolo_conf,
            max_skip_streak=args.max_skip_streak,
        )

        if not args.skip_kframe:
            stage_7_kframe_baseline(
                base_dir=test_base_dir,
                sequences=test_sequences,
                split_name="test",
                yolo_model=yolo_model,
                k=args.k,
                yolo_conf=args.yolo_conf,
            )

        stage_8_evaluate_adaptive_reduced(
            sequences=test_sequences,
            split_name="test",
            iou_thresh=args.iou_thresh,
            max_temporal_gap=args.max_temporal_gap,
            match_class=True,
        )

        if not args.skip_kframe:
            stage_9_evaluate_kframe(
                sequences=test_sequences,
                split_name="test",
                k=args.k,
                iou_thresh=args.iou_thresh,
                max_temporal_gap=args.max_temporal_gap,
                match_class=True,
            )

    print("\n" + "=" * 90)
    print("DONE: Full DPM + FRCNN + SDP project pipeline completed successfully.")
    print("=" * 90)

    print("\nMain outputs:")
    print(f"  RF model: {PROJECT_ROOT / 'outputs/models/random_forest_all_variants/random_forest_gt_model.joblib'}")
    print(f"  RF metrics: {PROJECT_ROOT / 'outputs/models/random_forest_all_variants/metrics.json'}")
    print(f"  Full YOLO: {PROJECT_ROOT / 'outputs/full_yolo_baseline_gt/json'}")
    print(f"  Adaptive inference: {PROJECT_ROOT / 'outputs/adaptive_inference_gt'}")
    print(f"  K-frame baseline: {PROJECT_ROOT / 'outputs/kframe_yolo_gt'}")
    print(f"  Adaptive-vs-GT eval: {PROJECT_ROOT / 'outputs/evaluation/adaptive_vs_gt_all_variants'}")


if __name__ == "__main__":
    main()