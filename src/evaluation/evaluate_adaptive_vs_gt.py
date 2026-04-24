from pathlib import Path
import json
import pandas as pd
import numpy as np


def compute_iou(box_a, box_b):
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


def greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5):
    matches = 0
    used_gt = set()
    used_pred = set()

    pairs = []
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = compute_iou(gt_box, pred_box)
            if iou >= iou_thresh:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matches += 1

    return matches, len(gt_boxes) - matches, len(pred_boxes) - matches


def evaluate_sequence(gt_csv, pred_csv, decisions_csv, iou_thresh=0.5):
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)
    decisions_df = pd.read_csv(decisions_csv)

    # GT in MOT17 is pedestrian-oriented; if class_id exists, keep class 1 for pedestrian labels if needed.
    # If your GT CSV includes all GT rows already, keep them all for now.
    # Predictions: only person class from COCO = 0
    pred_df = pred_df[pred_df["class_id"] == 0].copy()

    frames = sorted(set(gt_df["frame"].unique()).union(set(pred_df["frame"].unique())))

    total_tp = 0
    total_fn = 0
    total_fp = 0

    per_frame_rows = []

    for frame_id in frames:
        gt_frame = gt_df[gt_df["frame"] == frame_id]
        pred_frame = pred_df[pred_df["frame"] == frame_id]

        gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].to_numpy().tolist()
        pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].to_numpy().tolist()

        tp, fn, fp = greedy_match(gt_boxes, pred_boxes, iou_thresh=iou_thresh)

        total_tp += tp
        total_fn += fn
        total_fp += fp

        per_frame_rows.append(
            {
                "frame": int(frame_id),
                "tp": int(tp),
                "fn": int(fn),
                "fp": int(fp),
                "num_gt": int(len(gt_boxes)),
                "num_pred": int(len(pred_boxes)),
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    yolo_run_rate = float(decisions_df["yolo_ran"].mean())
    frame_reduction = 1.0 - yolo_run_rate

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(total_tp),
        "fn": int(total_fn),
        "fp": int(total_fp),
        "yolo_run_rate": yolo_run_rate,
        "frame_reduction": frame_reduction,
        "num_frames": int(len(frames)),
    }

    return metrics, pd.DataFrame(per_frame_rows)


def main():
    gt_dir = Path("data/interim/gt_annotations")
    pred_dir = Path("outputs/adaptive_inference_gt/detections")
    decisions_dir = Path("outputs/adaptive_inference_gt/decisions")
    output_dir = Path("outputs/evaluation/adaptive_vs_gt")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    all_metrics = {}

    for seq_name in sequences:
        gt_csv = gt_dir / f"{seq_name}_gt.csv"
        pred_csv = pred_dir / f"{seq_name}_adaptive_detections.csv"
        decisions_csv = decisions_dir / f"{seq_name}_decisions.csv"

        metrics, per_frame_df = evaluate_sequence(
            gt_csv=gt_csv,
            pred_csv=pred_csv,
            decisions_csv=decisions_csv,
            iou_thresh=0.5,
        )

        all_metrics[seq_name] = metrics

        with open(output_dir / f"{seq_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        per_frame_df.to_csv(output_dir / f"{seq_name}_per_frame.csv", index=False)

        print(f"\n=== {seq_name} ===")
        print(json.dumps(metrics, indent=2))

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":
    main()