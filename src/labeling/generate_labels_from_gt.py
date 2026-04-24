from pathlib import Path
import math
import pandas as pd


def compute_box_center(row):
    cx = (row["x1"] + row["x2"]) / 2.0
    cy = (row["y1"] + row["y2"]) / 2.0
    return cx, cy


def compute_center_distance(row_prev, row_curr):
    cx_prev, cy_prev = compute_box_center(row_prev)
    cx_curr, cy_curr = compute_box_center(row_curr)

    dx = cx_curr - cx_prev
    dy = cy_curr - cy_prev
    return math.sqrt(dx * dx + dy * dy)


def generate_labels_for_sequence_from_gt(
    gt_csv: Path,
    output_csv: Path,
    motion_threshold: float = 20.0,
):
    """
    Generate frame-level keep/skip labels from MOT17 ground-truth annotations.

    keep = 1 if:
      - count changed
      - any ID appeared
      - any ID disappeared
      - any shared ID moved significantly
    else keep = 0
    """
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")

    df = pd.read_csv(gt_csv)

    required_cols = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {gt_csv}: {missing}")

    frames = sorted(df["frame"].unique())
    if not frames:
        raise ValueError(f"No frames found in {gt_csv}")

    rows = []

    # First frame is always keep
    first_frame = frames[0]
    first_df = df[df["frame"] == first_frame].copy()
    first_ids = set(first_df["track_id"].astype(int).tolist())

    rows.append(
        {
            "frame": first_frame,
            "label": 1,
            "prev_count": 0,
            "curr_count": len(first_ids),
            "num_common_ids": 0,
            "num_appeared": len(first_ids),
            "num_disappeared": 0,
            "count_changed": int(len(first_ids) > 0),
            "max_center_displacement": 0.0,
            "motion_flag": 0,
        }
    )

    for prev_frame, curr_frame in zip(frames[:-1], frames[1:]):
        prev_df = df[df["frame"] == prev_frame].copy()
        curr_df = df[df["frame"] == curr_frame].copy()

        prev_df["track_id"] = prev_df["track_id"].astype(int)
        curr_df["track_id"] = curr_df["track_id"].astype(int)

        prev_ids = set(prev_df["track_id"].tolist())
        curr_ids = set(curr_df["track_id"].tolist())

        common_ids = prev_ids & curr_ids
        appeared_ids = curr_ids - prev_ids
        disappeared_ids = prev_ids - curr_ids

        prev_count = len(prev_ids)
        curr_count = len(curr_ids)
        count_changed = int(prev_count != curr_count)

        max_center_displacement = 0.0

        for track_id in common_ids:
            prev_row = prev_df[prev_df["track_id"] == track_id].iloc[0]
            curr_row = curr_df[curr_df["track_id"] == track_id].iloc[0]

            dist = compute_center_distance(prev_row, curr_row)
            if dist > max_center_displacement:
                max_center_displacement = dist

        motion_flag = int(max_center_displacement > motion_threshold)

        keep = int(
            count_changed
            or len(appeared_ids) > 0
            or len(disappeared_ids) > 0
            or motion_flag > 0
        )

        # NOTE: tune label generation here
        significant_count_change = int(abs(curr_count - prev_count) >= 1)
        significant_appearance = int(len(appeared_ids) >= 1)
        significant_disappearance = int(len(disappeared_ids) >= 1)

        keep = int(
            significant_count_change
            or significant_appearance
            or significant_disappearance
            or motion_flag > 0
        )

        # rows.append(
        #     {
        #         "frame": curr_frame,
        #         "label": keep,
        #         "prev_count": prev_count,
        #         "curr_count": curr_count,
        #         "num_common_ids": len(common_ids),
        #         "num_appeared": len(appeared_ids),
        #         "num_disappeared": len(disappeared_ids),
        #         "count_changed": count_changed,
        #         "max_center_displacement": max_center_displacement,
        #         "motion_flag": motion_flag,
        #     }
        # )
        
        rows.append(
            {
                "frame": curr_frame,
                "label": keep,
                "prev_count": prev_count,
                "curr_count": curr_count,
                "num_common_ids": len(common_ids),
                "num_appeared": len(appeared_ids),
                "num_disappeared": len(disappeared_ids),
                "count_changed": count_changed,
                "significant_count_change": significant_count_change,
                "significant_appearance": significant_appearance,
                "significant_disappearance": significant_disappearance,
                "max_center_displacement": max_center_displacement,
                "motion_flag": motion_flag,
            }
        )

    labels_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_csv, index=False)

    print(f"Saved GT-based labels to {output_csv}")


def main():
    gt_dir = Path("data/interim/gt_annotations")
    output_dir = Path("data/interim/labels_gt")

    # sequences = [
    #     "MOT17-09-SDP", # NOTE: Ok
    #     # "MOT17-10-SDP", # NOTE: Not good
    #     # "MOT17-11-SDP", # NOTE: Good
    # ]

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
        gt_csv = gt_dir / f"{seq_name}_gt.csv"
        output_csv = output_dir / f"{seq_name}_labels.csv"

        generate_labels_for_sequence_from_gt(
            gt_csv=gt_csv,
            output_csv=output_csv,
            motion_threshold=20.0, # NOTE: tune parameter here
        )


if __name__ == "__main__":
    main()