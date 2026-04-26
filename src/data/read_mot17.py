from pathlib import Path
import pandas as pd
from visualize_mot17 import visualize_gt


GT_COLUMNS = [
    "frame",
    "track_id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "class_id",
    "visibility",
]


def read_mot17_gt_file(path: Path):
    """
    Read MOT17 ground-truth file and return a clean dataframe.
    """
    if not path.exists():
        raise FileNotFoundError(f"GT file not found: {path}")

    df = pd.read_csv(path, header=None, names=GT_COLUMNS)

    # convert to xyxy box format
    df["x1"] = df["bb_left"]
    df["y1"] = df["bb_top"]
    df["x2"] = df["bb_left"] + df["bb_width"]
    df["y2"] = df["bb_top"] + df["bb_height"]

    df = df[
        [
            "frame",
            "track_id",
            "class_id",
            "visibility",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "bb_width",
            "bb_height",
        ]
    ].copy()

    return df


def save_gt_for_sequence(sequence_dir: Path, output_csv: Path):
    gt_path = sequence_dir / "gt" / "gt.txt"
    df = read_mot17_gt_file(gt_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved GT annotations to {output_csv}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    data_output_dir = Path("data/interim/gt_annotations")
    visual_output_dir = Path("data/interim/gt_visualizations")

    sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ]

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name
        gt_csv = data_output_dir / f"{seq_name}_gt.csv"
        save_gt_for_sequence(sequence_dir, gt_csv)
        visualize_gt(
            sequence_dir=sequence_dir,
            gt_csv=gt_csv,
            output_dir=visual_output_dir / seq_name,
        )


if __name__ == "__main__":
    main()