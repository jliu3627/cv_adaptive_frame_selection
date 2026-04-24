import json
import argparse
from pathlib import Path
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_dets(raw):
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}

    if isinstance(raw, list):
        return {
            int(item["frame_id"]): item.get("detections", [])
            for item in raw
        }

    raise ValueError("Unsupported detection JSON format.")


def match_boxes(base_dets, pred_dets, iou_thresh=0.5, match_class=True):
    tp, fn = 0, 0
    matched_pred = set()
    conf_drops = []

    for base in base_dets:
        best_iou = 0
        best_j = None

        for j, pred in enumerate(pred_dets):
            if j in matched_pred:
                continue

            if match_class and base.get("cls") != pred.get("cls"):
                continue

            score = iou(base["bbox"], pred["bbox"])

            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou >= iou_thresh:
            tp += 1
            matched_pred.add(best_j)

            if "conf" in base and "conf" in pred_dets[best_j]:
                conf_drops.append(base["conf"] - pred_dets[best_j]["conf"])
        else:
            fn += 1

    fp = len(pred_dets) - len(matched_pred)
    return tp, fp, fn, conf_drops


def summarize(tp, fp, fn, conf_drops):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "avg_confidence_drop": float(np.mean(conf_drops)) if conf_drops else None,
    }


def evaluate_keep_frames_only(baseline, reduced, kept_frames, iou_thresh, match_class):
    tp = fp = fn = 0
    conf_drops = []

    for frame_id in sorted(kept_frames):
        base = baseline.get(frame_id, [])
        pred = reduced.get(frame_id, [])

        t, f, n, drops = match_boxes(base, pred, iou_thresh, match_class)

        tp += t
        fp += f
        fn += n
        conf_drops.extend(drops)

    return summarize(tp, fp, fn, conf_drops)


def nearest_kept(frame_id, kept_frames):
    kept = np.array(sorted(kept_frames))
    idx = np.searchsorted(kept, frame_id)

    prev_frame = int(kept[idx - 1]) if idx > 0 else None
    next_frame = int(kept[idx]) if idx < len(kept) else None

    return prev_frame, next_frame


def evaluate_adaptive_temporal(
    baseline,
    reduced,
    kept_frames,
    iou_thresh,
    match_class,
    max_temporal_gap,
):
    tp = fp = fn = 0
    conf_drops = []
    used_preds = set()

    kept_frames = set(kept_frames)

    for frame_id in sorted(baseline.keys()):
        base_dets = baseline.get(frame_id, [])

        if not base_dets:
            continue

        candidate_frames = []

        if frame_id in kept_frames:
            candidate_frames.append(frame_id)
        else:
            prev_frame, next_frame = nearest_kept(frame_id, kept_frames)

            if prev_frame is not None:
                if abs(frame_id - prev_frame) <= max_temporal_gap:
                    candidate_frames.append(prev_frame)

            if next_frame is not None:
                if abs(next_frame - frame_id) <= max_temporal_gap:
                    candidate_frames.append(next_frame)

        candidate_preds = []

        for cf in candidate_frames:
            for det_idx, det in enumerate(reduced.get(cf, [])):
                candidate_preds.append({
                    **det,
                    "_frame": cf,
                    "_idx": det_idx,
                })

        for base in base_dets:
            best_iou = 0
            best_key = None
            best_pred = None

            for pred in candidate_preds:
                key = (pred["_frame"], pred["_idx"])

                if key in used_preds:
                    continue

                if match_class and base.get("cls") != pred.get("cls"):
                    continue

                score = iou(base["bbox"], pred["bbox"])

                if score > best_iou:
                    best_iou = score
                    best_key = key
                    best_pred = pred

            if best_iou >= iou_thresh:
                tp += 1
                used_preds.add(best_key)

                if "conf" in base and "conf" in best_pred:
                    conf_drops.append(base["conf"] - best_pred["conf"])
            else:
                fn += 1

    for frame_id, dets in reduced.items():
        for det_idx, _ in enumerate(dets):
            if (frame_id, det_idx) not in used_preds:
                fp += 1

    return summarize(tp, fp, fn, conf_drops)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_json", required=True)
    parser.add_argument("--reduced_json", required=True)
    parser.add_argument("--kept_frames_json", required=True)
    parser.add_argument("--output_json", required=True)

    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--max_temporal_gap", type=int, default=10)
    parser.add_argument("--ignore_class", action="store_true")

    args = parser.parse_args()

    baseline = normalize_dets(load_json(args.baseline_json))
    reduced = normalize_dets(load_json(args.reduced_json))
    kept_frames = set(load_json(args.kept_frames_json))

    match_class = not args.ignore_class

    num_frames = max(baseline.keys()) + 1 if baseline else 0
    yolo_run_rate = len(kept_frames) / num_frames if num_frames else 0.0

    results = {
        "keep_frame_only": evaluate_keep_frames_only(
            baseline,
            reduced,
            kept_frames,
            args.iou_thresh,
            match_class,
        ),
        "adaptive_temporal": evaluate_adaptive_temporal(
            baseline,
            reduced,
            kept_frames,
            args.iou_thresh,
            match_class,
            args.max_temporal_gap,
        ),
        "num_frames": num_frames,
        "num_kept_frames": len(kept_frames),
        "yolo_run_rate": yolo_run_rate,
        "frame_reduction": 1.0 - yolo_run_rate,
        "iou_thresh": args.iou_thresh,
        "max_temporal_gap": args.max_temporal_gap,
    }

    save_json(results, args.output_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()