from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def draw_gt_boxes(image, frame_df):
    output = image.copy()

    for _, row in frame_df.iterrows():
        x1 = int(round(row["x1"]))
        y1 = int(round(row["y1"]))
        x2 = int(round(row["x2"]))
        y2 = int(round(row["y2"]))
        track_id = int(row["track_id"])
        class_id = int(row["class_id"])

        label = f"id {track_id} | class {class_id}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        text_x = x1
        text_y = max(y1 - 8, text_h + 4)

        cv2.rectangle(
            output,
            (text_x, text_y - text_h - 4),
            (text_x + text_w + 4, text_y + baseline - 4),
            (0, 255, 0),
            thickness=-1,
        )

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


def visualize_gt_for_sequence(
    sequence_dir: Path,
    gt_csv: Path,
    output_dir: Path,
    max_frames: int | None = None,
):
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")

    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")

    df = pd.read_csv(gt_csv)

    image_paths = sorted(img_dir.glob("*.jpg"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc=f"GT vis {sequence_dir.name}"):
        frame_id = int(img_path.stem)

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        frame_df = df[df["frame"] == frame_id]
        vis_image = draw_gt_boxes(image, frame_df)

        cv2.putText(
            vis_image,
            f"Frame: {frame_id:06d} | GT boxes: {len(frame_df)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), vis_image)

    print(f"Saved GT visualizations to {output_dir}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    gt_dir = Path("data/interim/gt_annotations")
    output_base = Path("data/interim/gt_visualizations")

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
        output_dir = output_base / seq_name

        visualize_gt_for_sequence(
            sequence_dir=sequence_dir,
            gt_csv=gt_csv,
            output_dir=output_dir,
            max_frames=None,
        )


if __name__ == "__main__":
    main()