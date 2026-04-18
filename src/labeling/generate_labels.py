from pathlib import Path
import numpy as np
import pandas as pd


def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes in xyxy format.
    box = [x1, y1, x2, y2]
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def get_boxes_for_frame(df, frame_id, confidence_thresh=0.4, person_only=True):
    """
    Return filtered boxes for a given frame.
    """
    frame_df = df[df["frame"] == frame_id].copy()

    if person_only:
        frame_df = frame_df[frame_df["class_id"] == 0]

    frame_df = frame_df[frame_df["confidence"] >= confidence_thresh]

    boxes = frame_df[["x1", "y1", "x2", "y2"]].to_numpy()
    return boxes


def match_boxes(prev_boxes, curr_boxes, iou_match_thresh=0.3):
    """
    Greedy IoU matching between previous and current boxes.
    Returns:
        matches: list of (prev_idx, curr_idx, iou)
        unmatched_prev: set of unmatched prev indices
        unmatched_curr: set of unmatched curr indices
    """
    matches = []
    unmatched_prev = set(range(len(prev_boxes)))
    unmatched_curr = set(range(len(curr_boxes)))

    if len(prev_boxes) == 0 or len(curr_boxes) == 0:
        return matches, unmatched_prev, unmatched_curr

    iou_pairs = []
    for i, prev_box in enumerate(prev_boxes):
        for j, curr_box in enumerate(curr_boxes):
            iou = compute_iou(prev_box, curr_box)
            if iou >= iou_match_thresh:
                iou_pairs.append((iou, i, j))

    # sort highest IoU first
    iou_pairs.sort(reverse=True, key=lambda x: x[0])

    used_prev = set()
    used_curr = set()

    for iou, i, j in iou_pairs:
        if i in used_prev or j in used_curr:
            continue
        matches.append((i, j, iou))
        used_prev.add(i)
        used_curr.add(j)

    unmatched_prev -= used_prev
    unmatched_curr -= used_curr

    return matches, unmatched_prev, unmatched_curr


def label_frame(
    prev_boxes,
    curr_boxes,
    iou_match_thresh=0.3,
    motion_iou_thresh=0.7,
):
    """
    Return frame-level label and debug info.
    """
    prev_count = len(prev_boxes)
    curr_count = len(curr_boxes)

    matches, unmatched_prev, unmatched_curr = match_boxes(
        prev_boxes, curr_boxes, iou_match_thresh=iou_match_thresh
    )

    num_appeared = len(unmatched_curr)
    num_disappeared = len(unmatched_prev)

    motion_flag = 0
    for _, _, iou in matches:
        if iou < motion_iou_thresh:
            motion_flag = 1
            break

    count_changed = int(prev_count != curr_count)

    keep = int(
        count_changed
        or num_appeared > 0
        or num_disappeared > 0
        or motion_flag > 0
    )

    debug = {
        "prev_count": prev_count,
        "curr_count": curr_count,
        "num_matches": len(matches),
        "num_appeared": num_appeared,
        "num_disappeared": num_disappeared,
        "motion_flag": motion_flag,
        "count_changed": count_changed,
    }

    return keep, debug


def generate_labels_for_sequence(
    detections_csv: Path,
    output_csv: Path,
    confidence_thresh=0.4,
    iou_match_thresh=0.3,
    motion_iou_thresh=0.7,
    person_only=True,
):
    df = pd.read_csv(detections_csv)

    frames = sorted(df["frame"].unique())
    if not frames:
        raise ValueError(f"No frames found in {detections_csv}")

    rows = []

    # First frame should always be kept
    first_frame = frames[0]
    first_boxes = get_boxes_for_frame(
        df, first_frame, confidence_thresh=confidence_thresh, person_only=person_only
    )

    rows.append(
        {
            "frame": first_frame,
            "label": 1,
            "prev_count": 0,
            "curr_count": len(first_boxes),
            "num_matches": 0,
            "num_appeared": len(first_boxes),
            "num_disappeared": 0,
            "motion_flag": 0,
            "count_changed": int(len(first_boxes) > 0),
        }
    )

    prev_boxes = first_boxes

    for frame_id in frames[1:]:
        curr_boxes = get_boxes_for_frame(
            df, frame_id, confidence_thresh=confidence_thresh, person_only=person_only
        )

        keep, debug = label_frame(
            prev_boxes,
            curr_boxes,
            iou_match_thresh=iou_match_thresh,
            motion_iou_thresh=motion_iou_thresh,
        )

        rows.append(
            {
                "frame": frame_id,
                "label": keep,
                **debug,
            }
        )

        prev_boxes = curr_boxes

    labels_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_csv, index=False)
    print(f"Saved labels to {output_csv}")


def main():
    detections_dir = Path("data/interim/detections")
    output_dir = Path("data/interim/labels")

    sequences = [
        "MOT17-10-SDP",
        "MOT17-11-SDP",
    ]

    for seq_name in sequences:
        detections_csv = detections_dir / f"{seq_name}.csv"
        output_csv = output_dir / f"{seq_name}_labels.csv"

        generate_labels_for_sequence(
            detections_csv=detections_csv,
            output_csv=output_csv,
            confidence_thresh=0.4,
            iou_match_thresh=0.3,
            motion_iou_thresh=0.7,
            person_only=True,
        )


if __name__ == "__main__":
    main()