"""
This script runs the latest trained Random Forest adaptive frame-selection model with in retraining.
"""

from __future__ import annotations

import argparse

from main import (
    PROJECT_ROOT,
    YOLO,
    build_sequence_names,
    discover_sequences,
    ensure_exists,
    print_step,
    stage_1_read_gt,
    stage_5_full_yolo_baseline,
    stage_6_adaptive_inference,
    stage_8_evaluate_adaptive_reduced,
    stage_10_evaluate_adaptive_vs_gt,
)


DEFAULT_SEQUENCE_IDS = ["02"]
DEFAULT_VARIANTS = ["SDP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run latest trained adaptive frame-selection model without retraining."
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=["DPM", "FRCNN", "SDP"],
        help="MOT17 variants to run.",
    )

    parser.add_argument(
        "--sequence-ids",
        nargs="+",
        default=DEFAULT_SEQUENCE_IDS,
        help="MOT17 sequence IDs to run. Default is 02 for a fast demo.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/random_forest/random_forest_gt_model.joblib",
        help="Path to saved Random Forest model.",
    )

    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--adaptive-keep-threshold",
        type=float,
        default=0.45,
        help="RF keep-probability threshold.",
    )

    parser.add_argument(
        "--max-skip-streak",
        type=int,
        default=5,
        help="Maximum skipped-frame streak before forced YOLO refresh.",
    )

    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation.",
    )

    parser.add_argument(
        "--max-temporal-gap",
        type=int,
        default=5,
        help="Maximum temporal gap for reduced detection evaluation.",
    )

    parser.add_argument(
        "--skip-gt-eval",
        action="store_true",
        help="Skip adaptive-vs-GT evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_base_dir = PROJECT_ROOT / "data/raw/MOT17/train"
    model_path = PROJECT_ROOT / args.model_path

    ensure_exists(train_base_dir, "MOT17 train folder")
    ensure_exists(model_path, "latest trained Random Forest model")

    variants = [variant.upper() for variant in args.variants]

    expected_sequences = build_sequence_names(
        sequence_ids=args.sequence_ids,
        variants=variants,
    )

    train_sequences = [
        seq_name
        for seq_name in expected_sequences
        if (train_base_dir / seq_name).exists()
    ]

    if not train_sequences:
        train_sequences = discover_sequences(
            base_dir=train_base_dir,
            variants=variants,
        )

    if not train_sequences:
        raise FileNotFoundError(
            "No MOT17 train sequences found.\n"
            "Expected folders like:\n"
            "  data/raw/MOT17/train/MOT17-02-SDP\n"
            "  data/raw/MOT17/train/MOT17-02-FRCNN\n"
            "  data/raw/MOT17/train/MOT17-02-DPM\n"
        )

    print_step("Final Submission Runner")
    print("Running latest trained model WITHOUT retraining.")
    print(f"\nModel path:\n  {model_path}")

    print("\nSelected sequences:")
    for seq_name in train_sequences:
        print(f"  - {seq_name}")

    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Install dependencies using:\n"
            "  pip install -r requirements.txt"
        )

    # GT CSV generation is only for evaluation, not model training.
    stage_1_read_gt(
        train_base_dir=train_base_dir,
        train_sequences=train_sequences,
    )

    print_step("Loading YOLO model")
    yolo_model = YOLO("yolov8n.pt")

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

    stage_8_evaluate_adaptive_reduced(
        sequences=train_sequences,
        split_name="train",
        iou_thresh=args.iou_thresh,
        max_temporal_gap=args.max_temporal_gap,
        match_class=True,
    )

    if not args.skip_gt_eval:
        stage_10_evaluate_adaptive_vs_gt(
            train_sequences=train_sequences,
            iou_thresh=args.iou_thresh,
        )

    print("\n" + "=" * 90)
    print("DONE: Latest trained model ran successfully without retraining.")
    print("=" * 90)

    print("\nGenerated outputs:")
    print(f"  Full YOLO baseline: {PROJECT_ROOT / 'outputs/full_yolo_baseline_gt/json'}")
    print(f"  Adaptive inference: {PROJECT_ROOT / 'outputs/adaptive_inference_gt'}")
    print(f"  Reduced eval: {PROJECT_ROOT / 'outputs/adaptive_inference_gt/reduced_detection_eval'}")
    print(f"  GT eval: {PROJECT_ROOT / 'outputs/evaluation/adaptive_vs_gt_all_variants'}")


if __name__ == "__main__":
    main()