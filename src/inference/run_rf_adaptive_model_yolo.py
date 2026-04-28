from pathlib import Path
import json
import joblib
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


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


def load_grayscale(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_frame_difference(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    return {
        "frame_diff_mean": float(np.mean(diff)),
        "frame_diff_max": float(np.max(diff)),
    }


def compute_optical_flow_features(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return {
        "flow_mean": float(np.mean(mag)),
        "flow_max": float(np.max(mag)),
        "flow_std": float(np.std(mag)),
    }


def compute_edge_difference(prev_gray, curr_gray):
    prev_edges = cv2.Canny(prev_gray, 100, 200)
    curr_edges = cv2.Canny(curr_gray, 100, 200)
    edge_diff = cv2.absdiff(prev_edges, curr_edges)
    return {
        "edge_diff_mean": float(np.mean(edge_diff) / 255.0),
        "edge_density_prev": float(np.mean(prev_edges) / 255.0),
        "edge_density_curr": float(np.mean(curr_edges) / 255.0),
    }


def compute_histogram_difference(prev_gray, curr_gray):
    prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])

    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()

    return {
        "hist_bhattacharyya": float(
            cv2.compareHist(
                prev_hist.astype(np.float32),
                curr_hist.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        ),
        "hist_corr": float(
            cv2.compareHist(
                prev_hist.astype(np.float32),
                curr_hist.astype(np.float32),
                cv2.HISTCMP_CORREL,
            )
        ),
    }


def extract_features_for_pair(prev_img_path: Path, curr_img_path: Path):
    prev_gray = load_grayscale(prev_img_path)
    curr_gray = load_grayscale(curr_img_path)

    features = {}
    features.update(compute_frame_difference(prev_gray, curr_gray))
    features.update(compute_optical_flow_features(prev_gray, curr_gray))
    features.update(compute_edge_difference(prev_gray, curr_gray))
    features.update(compute_histogram_difference(prev_gray, curr_gray))
    return features


def run_yolo_on_frame(model, image_path: Path, confidence_thresh=0.25):
    results = model(str(image_path), verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    detections = []

    for box, conf, cls_id in zip(xyxy, confs, classes):
        conf = float(conf)

        if conf < confidence_thresh:
            continue

        x1, y1, x2, y2 = box

        detections.append(
            {
                # JSON format for new evaluator
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": conf,
                "cls": int(cls_id),

                # CSV-friendly fields
                "class_id": int(cls_id),
                "confidence": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    return detections


def save_detection_rows(frame_id, detections, source):
    rows = []

    for det in detections:
        rows.append(
            {
                "frame": frame_id,
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "source": source,
            }
        )

    return rows


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def clean_detections_for_json(detections):
    """
    Removes duplicate CSV-only fields and keeps evaluator format.
    """
    clean = []

    for det in detections:
        clean.append(
            {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "cls": det["cls"],
            }
        )

    return clean


def run_adaptive_inference_for_sequence(
    sequence_dir: Path,
    rf_model,
    yolo_model,
    decisions_output_csv: Path,
    detections_output_csv: Path,
    reduced_json_output: Path,
    kept_frames_output: Path,
    keep_threshold=0.45,
    yolo_conf_thresh=0.25,
    max_skip_streak=5,
):
    img_dir = sequence_dir / "img1"
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No images found in {img_dir}")

    decision_rows = []
    detection_rows = []

    # NEW: these are for the reduced-frame evaluation
    reduced_yolo_detections = {}
    kept_frames = []

    prev_img_path = None
    last_detections = []
    skip_streak = 0

    for idx, curr_img_path in enumerate(
        tqdm(image_paths, desc=f"Adaptive {sequence_dir.name}")
    ):
        frame_id = int(curr_img_path.stem)

        if idx == 0:
            detections = run_yolo_on_frame(
                yolo_model,
                curr_img_path,
                confidence_thresh=yolo_conf_thresh,
            )

            last_detections = detections
            skip_streak = 0

            kept_frames.append(frame_id)
            reduced_yolo_detections[frame_id] = clean_detections_for_json(detections)

            decision_rows.append(
                {
                    "frame": frame_id,
                    "pred_label": 1,
                    "keep_probability": 1.0,
                    "yolo_ran": 1,
                    "forced_refresh": 1,
                    "skip_streak_before": 0,
                    "num_detections": len(detections),
                }
            )

            detection_rows.extend(save_detection_rows(frame_id, detections, "yolo"))

            prev_img_path = curr_img_path
            continue

        features = extract_features_for_pair(prev_img_path, curr_img_path)
        X_curr = pd.DataFrame([features])[FEATURE_COLS]

        keep_prob = float(rf_model.predict_proba(X_curr)[0][1])
        pred_label = int(keep_prob >= keep_threshold)

        forced_refresh = 0
        skip_streak_before = skip_streak

        if skip_streak >= max_skip_streak:
            pred_label = 1
            forced_refresh = 1

        if pred_label == 1:
            detections = run_yolo_on_frame(
                yolo_model,
                curr_img_path,
                confidence_thresh=yolo_conf_thresh,
            )

            last_detections = detections
            yolo_ran = 1
            source = "yolo"
            skip_streak = 0

            # NEW: only real YOLO-run frames are kept here
            kept_frames.append(frame_id)
            reduced_yolo_detections[frame_id] = clean_detections_for_json(detections)

        else:
            # Reused detections are still saved to CSV for visualization/debugging,
            # but NOT saved as YOLO detections for reduced-frame evaluation.
            detections = last_detections
            yolo_ran = 0
            source = "reused"
            skip_streak += 1

            # NEW: skipped frame gets empty YOLO detections
            reduced_yolo_detections[frame_id] = []

        decision_rows.append(
            {
                "frame": frame_id,
                "pred_label": pred_label,
                "keep_probability": keep_prob,
                "yolo_ran": yolo_ran,
                "forced_refresh": forced_refresh,
                "skip_streak_before": skip_streak_before,
                "num_detections": len(detections),
            }
        )

        detection_rows.extend(save_detection_rows(frame_id, detections, source))

        prev_img_path = curr_img_path

    decisions_df = pd.DataFrame(decision_rows)
    detections_df = pd.DataFrame(detection_rows)

    decisions_output_csv.parent.mkdir(parents=True, exist_ok=True)
    detections_output_csv.parent.mkdir(parents=True, exist_ok=True)

    decisions_df.to_csv(decisions_output_csv, index=False)
    detections_df.to_csv(detections_output_csv, index=False)

    save_json(reduced_yolo_detections, reduced_json_output)
    save_json(kept_frames, kept_frames_output)

    print(f"Saved decisions to {decisions_output_csv}")
    print(f"Saved detections CSV to {detections_output_csv}")
    print(f"Saved reduced YOLO JSON to {reduced_json_output}")
    print(f"Saved kept frames to {kept_frames_output}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    model_path = Path("outputs/models/random_forest/random_forest_gt_model.joblib")

    output_dir = Path("outputs/adaptive_inference_gt")

    decisions_dir = output_dir / "decisions"
    detections_dir = output_dir / "detections"

    # NEW
    reduced_json_dir = output_dir / "reduced_yolo_json"
    kept_frames_dir = output_dir / "kept_frames"

    sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ]

    rf_model = joblib.load(model_path)
    yolo_model = YOLO("yolov8n.pt")

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name

        decisions_output_csv = decisions_dir / f"{seq_name}_decisions.csv"
        detections_output_csv = detections_dir / f"{seq_name}_adaptive_detections.csv"

        # NEW
        reduced_json_output = reduced_json_dir / f"{seq_name}_reduced_yolo_detections.json"
        kept_frames_output = kept_frames_dir / f"{seq_name}_kept_frames.json"

        run_adaptive_inference_for_sequence(
            sequence_dir=sequence_dir,
            rf_model=rf_model,
            yolo_model=yolo_model,
            decisions_output_csv=decisions_output_csv,
            detections_output_csv=detections_output_csv,
            reduced_json_output=reduced_json_output,
            kept_frames_output=kept_frames_output,
            keep_threshold=0.45,
            yolo_conf_thresh=0.25,
            max_skip_streak=5,
        )

    # test dataset
    base_dir = Path("data/raw/MOT17/test")
    model_path = Path("outputs/models/random_forest/random_forest_gt_model.joblib")

    output_dir = Path("outputs/test/adaptive_inference")

    decisions_dir = output_dir / "decisions"
    detections_dir = output_dir / "detections"
    reduced_json_dir = output_dir / "reduced_yolo_json"
    kept_frames_dir = output_dir / "kept_frames"

    sequences = sorted([p.name for p in base_dir.iterdir() if p.is_dir() if "FRCNN" in p.name])

    rf_model = joblib.load(model_path)
    yolo_model = YOLO("yolov8n.pt")

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name

        decisions_output_csv = decisions_dir / f"{seq_name}_decisions.csv"
        detections_output_csv = detections_dir / f"{seq_name}_adaptive_detections.csv"
        reduced_json_output = reduced_json_dir / f"{seq_name}_reduced_yolo_detections.json"
        kept_frames_output = kept_frames_dir / f"{seq_name}_kept_frames.json"

        run_adaptive_inference_for_sequence(
            sequence_dir=sequence_dir,
            rf_model=rf_model,
            yolo_model=yolo_model,
            decisions_output_csv=decisions_output_csv,
            detections_output_csv=detections_output_csv,
            reduced_json_output=reduced_json_output,
            kept_frames_output=kept_frames_output,
            keep_threshold=0.45,
            yolo_conf_thresh=0.25,
            max_skip_streak=5,
        )

if __name__ == "__main__":
    main()