from pathlib import Path
import pandas as pd


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


def read_mot17_gt_file(gt_path: Path) -> pd.DataFrame:
    """
    Read MOT17 ground-truth file and return a clean dataframe.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    df = pd.read_csv(gt_path, header=None, names=GT_COLUMNS)

    # Convert to xyxy box format
    df["x1"] = df["bb_left"]
    df["y1"] = df["bb_top"]
    df["x2"] = df["bb_left"] + df["bb_width"]
    df["y2"] = df["bb_top"] + df["bb_height"]

    # Keep useful columns
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


def save_gt_for_sequence(sequence_dir: Path, output_csv: Path) -> None:
    gt_path = sequence_dir / "gt" / "gt.txt"
    df = read_mot17_gt_file(gt_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved GT annotations to {output_csv}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    output_dir = Path("data/interim/gt_annotations")

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
        output_csv = output_dir / f"{seq_name}_gt.csv"
        save_gt_for_sequence(sequence_dir, output_csv)


if __name__ == "__main__":
    main()