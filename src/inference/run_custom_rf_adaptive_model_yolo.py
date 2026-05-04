from pathlib import Path

import joblib
from ultralytics import YOLO

from run_rf_adaptive_model_yolo import (
    load_high_motion_thresholds,
    run_adaptive_inference_for_sequence,
)


def main():
    # Create this sequence with:
    # python src/data/make_custom_motion_sequence.py --overwrite
    custom_sequence_dir = Path("data/processed/custom_sequences/MOT17-custom-lowhigh")
    model_path = Path("outputs/models/random_forest/random_forest_gt_model.joblib")
    high_motion_thresholds_path = Path("outputs/models/random_forest/high_motion_thresholds.json")
    output_dir = Path("outputs/custom/adaptive_inference")

    rf_model = joblib.load(model_path)
    high_motion_thresholds = load_high_motion_thresholds(high_motion_thresholds_path)
    yolo_model = YOLO("yolov8n.pt")

    seq_name = custom_sequence_dir.name
    run_adaptive_inference_for_sequence(
        sequence_dir=custom_sequence_dir,
        rf_model=rf_model,
        yolo_model=yolo_model,
        decisions_output_csv=output_dir / "decisions" / f"{seq_name}_decisions.csv",
        detections_output_csv=output_dir / "detections" / f"{seq_name}_adaptive_detections.csv",
        reduced_json_output=output_dir / "reduced_yolo_json" / f"{seq_name}_reduced_yolo_detections.json",
        kept_frames_output=output_dir / "kept_frames" / f"{seq_name}_kept_frames.json",
        keep_threshold=0.45,
        yolo_conf_thresh=0.25,
        max_skip_streak=1000,
        high_motion_thresholds=high_motion_thresholds,
    )


if __name__ == "__main__":
    main()
