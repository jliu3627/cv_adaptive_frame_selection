from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_grayscale(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def compute_frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    return {
        "frame_diff_mean": float(np.mean(diff)),
        "frame_diff_max": float(np.max(diff)),
        "frame_diff_p90": float(np.percentile(diff, 90)),
        "frame_diff_p95": float(np.percentile(diff, 95)),
        "frame_diff_p99": float(np.percentile(diff, 99)),
        "changed_pixel_ratio_25": float(np.mean(diff > 25)),
        "changed_pixel_ratio_50": float(np.mean(diff > 50)),
    }


def compute_optical_flow_features(prev_gray: np.ndarray, curr_gray: np.ndarray):
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
        "flow_p90": float(np.percentile(mag, 90)),
        "flow_p95": float(np.percentile(mag, 95)),
        "flow_p99": float(np.percentile(mag, 99)),
        "flow_top10_mean": float(np.mean(mag[mag >= np.percentile(mag, 90)])),
    }


def compute_edge_difference(prev_gray: np.ndarray, curr_gray: np.ndarray):
    prev_edges = cv2.Canny(prev_gray, 100, 200)
    curr_edges = cv2.Canny(curr_gray, 100, 200)

    edge_diff = cv2.absdiff(prev_edges, curr_edges)

    return {
        "edge_diff_mean": float(np.mean(edge_diff) / 255.0),
        "edge_density_prev": float(np.mean(prev_edges) / 255.0),
        "edge_density_curr": float(np.mean(curr_edges) / 255.0),
    }


def compute_histogram_difference(prev_gray: np.ndarray, curr_gray: np.ndarray):
    prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])

    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()

    bhattacharyya = cv2.compareHist(
        prev_hist.astype(np.float32),
        curr_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )

    correlation = cv2.compareHist(
        prev_hist.astype(np.float32),
        curr_hist.astype(np.float32),
        cv2.HISTCMP_CORREL,
    )

    return {
        "hist_bhattacharyya": float(bhattacharyya),
        "hist_corr": float(correlation),
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


def prefix_features(features: dict, prefix: str):
    return {f"{prefix}{key}": value for key, value in features.items()}


def extract_features_for_sequence(
    sequence_dir: Path,
    labels_csv: Path,
    output_csv: Path,
) -> None:
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    labels_df = pd.read_csv(labels_csv)
    if "frame" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError(f"Labels CSV must contain 'frame' and 'label': {labels_csv}")

    # image sequence
    image_paths = sorted(img_dir.glob("*.jpg"))
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {img_dir}")

    # frame -> full label row lookup
    label_lookup = {
        int(row["frame"]): row.to_dict()
        for _, row in labels_df.iterrows()
    }

    rows = []
    last_keep_img_path = image_paths[0]
    frames_since_last_keep = 0

    # image features
    for i in tqdm(range(1, len(image_paths)), desc=f"Features {sequence_dir.name}"):
        prev_img_path = image_paths[i - 1]
        curr_img_path = image_paths[i]

        curr_frame = int(curr_img_path.stem)

        if curr_frame not in label_lookup:
            continue

        features = extract_features_for_pair(prev_img_path, curr_img_path)
        since_keep_features = prefix_features(
            extract_features_for_pair(last_keep_img_path, curr_img_path),
            "since_keep_",
        )
        label_row = label_lookup[curr_frame]
        label = int(label_row["label"])

        row = {
            "frame": curr_frame,
            **features,
            **since_keep_features,
            "frames_since_last_keep": frames_since_last_keep,
            "label": label,
        }

        # NOTE: columns from label generation
        debug_cols = [
            "prev_count",
            "curr_count",
            "num_common_ids",
            "num_appeared",
            "num_disappeared",
            "count_changed",
            "significant_count_change",
            "significant_appearance",
            "significant_disappearance",
            "max_center_displacement",
            "motion_flag",
        ]

        for col in debug_cols:
            if col in label_row:
                row[col] = label_row[col]

        rows.append(row)

        if label == 1:
            last_keep_img_path = curr_img_path
            frames_since_last_keep = 0
        else:
            frames_since_last_keep += 1

    features_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_csv, index=False)

    print(f"Saved features with GT labels to {output_csv}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    gt_labels_dir = Path("data/interim/gt_label")
    feature_output_dir = Path("data/processed/features")

    sequences = [Path(gt_csv).stem.split("_")[0] for gt_csv in gt_labels_dir.glob("*.csv")]

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name
        labels_csv = gt_labels_dir / f"{seq_name}_labels.csv"
        output_csv = feature_output_dir / f"{seq_name}_features.csv"

        extract_features_for_sequence(
            sequence_dir=sequence_dir,
            labels_csv=labels_csv,
            output_csv=output_csv,
        )


if __name__ == "__main__":
    main()
