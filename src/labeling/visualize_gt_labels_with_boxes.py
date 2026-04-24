from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def draw_gt_boxes(image, gt_df):
    """
    Draw GT boxes with track IDs.
    """
    output = image.copy()

    for _, row in gt_df.iterrows():
        x1 = int(round(row["x1"]))
        y1 = int(round(row["y1"]))
        x2 = int(round(row["x2"]))
        y2 = int(round(row["y2"]))
        track_id = int(row["track_id"])

        label = f"id {track_id}"

        # Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        text_x = x1
        text_y = max(y1 - 8, text_h + 4)

        # Text background
        cv2.rectangle(
            output,
            (text_x, text_y - text_h - 4),
            (text_x + text_w + 4, text_y + baseline - 4),
            (0, 255, 0),
            thickness=-1,
        )

        # Text
        cv2.putText(
            output,
            label,
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_label_overlay(image, row):
    """
    Draw frame-level KEEP/SKIP label and GT-based debug info.
    """
    output = image.copy()

    label_value = int(row["label"])
    label_text = "KEEP" if label_value == 1 else "SKIP"

    banner_color = (0, 180, 0) if label_value == 1 else (0, 0, 200)

    h, w = output.shape[:2]

    # Top banner
    cv2.rectangle(output, (0, 0), (w, 75), banner_color, thickness=-1)

    cv2.putText(
        output,
        f"LABEL: {label_text}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Frame: {int(row['frame']):06d}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    debug_cols = [
        "prev_count",
        "curr_count",
        "num_common_ids",
        "num_appeared",
        "num_disappeared",
        "count_changed",
        "max_center_displacement",
        "motion_flag",
    ]

    debug_lines = []
    for col in debug_cols:
        if col not in row.index:
            continue

        value = row[col]
        if col == "max_center_displacement":
            text = f"{col}: {float(value):.2f}"
        else:
            text = f"{col}: {value}"
        debug_lines.append(text)

    start_y = 105
    for i, text in enumerate(debug_lines):
        y = start_y + i * 28

        # black outline
        cv2.putText(
            output,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        # white text
        cv2.putText(
            output,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def visualize_sequence_with_gt_and_labels(
    sequence_dir: Path,
    gt_csv: Path,
    labels_csv: Path,
    output_dir: Path,
    max_frames: int | None = None,
):
    """
    Visualize original frames with:
    - GT boxes and track IDs
    - GT-based frame KEEP/SKIP labels
    """
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")

    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    gt_df = pd.read_csv(gt_csv)
    labels_df = pd.read_csv(labels_csv)

    required_gt_cols = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing_gt = required_gt_cols - set(gt_df.columns)
    if missing_gt:
        raise ValueError(f"Missing GT columns in {gt_csv}: {missing_gt}")

    required_label_cols = {"frame", "label"}
    missing_label = required_label_cols - set(labels_df.columns)
    if missing_label:
        raise ValueError(f"Missing label columns in {labels_csv}: {missing_label}")

    label_lookup = {int(row["frame"]): row for _, row in labels_df.iterrows()}

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc=f"GT label vis {sequence_dir.name}"):
        frame_id = int(img_path.stem)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: could not read image {img_path}")
            continue

        frame_gt = gt_df[gt_df["frame"] == frame_id]
        vis_image = draw_gt_boxes(image, frame_gt)

        if frame_id in label_lookup:
            vis_image = draw_label_overlay(vis_image, label_lookup[frame_id])
        else:
            cv2.putText(
                vis_image,
                f"Frame: {frame_id:06d} | NO LABEL",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Add quick GT count in corner
        count_text = f"GT boxes: {len(frame_gt)}"
        cv2.putText(
            vis_image,
            count_text,
            (15, vis_image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), vis_image)

    print(f"Saved GT label visualizations to {output_dir}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    gt_dir = Path("data/interim/gt_annotations")
    labels_dir = Path("data/interim/labels_gt")
    output_base = Path("data/interim/gt_label_visualizations")

    sequences = [
        "MOT17-02-DPM",
        "MOT17-02-FRCNN",
        "MOT17-02-SDP",
        "MOT17-04-DPM",
        "MOT17-04-FRCNN",
        "MOT17-04-SDP",
        "MOT17-05-DPM",
        "MOT17-05-FRCNN",
        "MOT17-05-SDP",
        "MOT17-09-DPM",
        "MOT17-09-FRCNN",
        "MOT17-09-SDP",
        "MOT17-10-DPM",
        "MOT17-10-FRCNN",
        "MOT17-10-SDP",
        "MOT17-11-DPM",
        "MOT17-11-FRCNN",
        "MOT17-11-SDP",
        "MOT17-13-DPM",
        "MOT17-13-FRCNN",
        "MOT17-13-SDP",
    ]

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name
        gt_csv = gt_dir / f"{seq_name}_gt.csv"
        labels_csv = labels_dir / f"{seq_name}_labels.csv"
        output_dir = output_base / seq_name

        visualize_sequence_with_gt_and_labels(
            sequence_dir=sequence_dir,
            gt_csv=gt_csv,
            labels_csv=labels_csv,
            output_dir=output_dir,
            max_frames=None,   # set to None for all frames
        )


if __name__ == "__main__":
    main()