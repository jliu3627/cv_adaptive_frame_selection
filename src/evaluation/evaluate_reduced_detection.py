from pathlib import Path
import json
import numpy as np


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_frame_dict(raw):
    return {int(k): v for k, v in raw.items()}


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(base_dets, pred_dets, iou_thresh=0.5, match_class=True):
    tp = 0
    fn = 0
    matched_pred = set()
    conf_drops = []

    for base in base_dets:
        best_iou = 0.0
        best_j = None

        for j, pred in enumerate(pred_dets):
            if j in matched_pred:
                continue

            if match_class and base.get("cls") != pred.get("cls"):
                continue

            score = box_iou(base["bbox"], pred["bbox"])

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
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "avg_confidence_drop": float(np.mean(conf_drops)) if conf_drops else None,
    }


def evaluate_keep_frames_only(
    baseline_dets,
    reduced_dets,
    kept_frames,
    iou_thresh=0.5,
    match_class=True,
):
    tp = fp = fn = 0
    conf_drops = []

    for frame_id in sorted(kept_frames):
        base = baseline_dets.get(frame_id, [])
        pred = reduced_dets.get(frame_id, [])

        t, f, n, drops = match_detections(
            base,
            pred,
            iou_thresh=iou_thresh,
            match_class=match_class,
        )

        tp += t
        fp += f
        fn += n
        conf_drops.extend(drops)

    return summarize(tp, fp, fn, conf_drops)


def nearest_kept_frames(frame_id, kept_frames_sorted):
    kept = np.array(kept_frames_sorted)
    idx = np.searchsorted(kept, frame_id)

    prev_frame = int(kept[idx - 1]) if idx > 0 else None
    next_frame = int(kept[idx]) if idx < len(kept) else None

    return prev_frame, next_frame


def evaluate_adaptive_temporal(
    baseline_dets,
    reduced_dets,
    kept_frames,
    iou_thresh=0.5,
    match_class=True,
    max_temporal_gap=5,
):
    kept_frames = set(kept_frames)
    kept_frames_sorted = sorted(kept_frames)

    tp = fp = fn = 0
    conf_drops = []

    # Track whether each reduced detection was used at least once
    # This is only for FP counting, not for blocking future matches.
    used_reduced_at_least_once = set()

    for frame_id in sorted(baseline_dets.keys()):
        base_dets = baseline_dets.get(frame_id, [])

        if not base_dets:
            continue

        candidate_frames = []

        if frame_id in kept_frames:
            candidate_frames.append(frame_id)
        else:
            prev_frame, next_frame = nearest_kept_frames(frame_id, kept_frames_sorted)

            if prev_frame is not None and abs(frame_id - prev_frame) <= max_temporal_gap:
                candidate_frames.append(prev_frame)

            if next_frame is not None and abs(next_frame - frame_id) <= max_temporal_gap:
                candidate_frames.append(next_frame)

        candidate_preds = []

        for candidate_frame in candidate_frames:
            for det_idx, det in enumerate(reduced_dets.get(candidate_frame, [])):
                candidate_preds.append(
                    {
                        **det,
                        "_source_frame": candidate_frame,
                        "_source_idx": det_idx,
                    }
                )

        # Important:
        # Only prevent duplicate matching within the SAME baseline frame.
        # Do NOT block reuse across different frames.
        used_within_this_frame = set()

        for base in base_dets:
            best_iou = 0.0
            best_key = None
            best_pred = None

            for pred in candidate_preds:
                key = (pred["_source_frame"], pred["_source_idx"])

                if key in used_within_this_frame:
                    continue

                if match_class and base.get("cls") != pred.get("cls"):
                    continue

                score = box_iou(base["bbox"], pred["bbox"])

                if score > best_iou:
                    best_iou = score
                    best_key = key
                    best_pred = pred

            if best_iou >= iou_thresh:
                tp += 1
                used_within_this_frame.add(best_key)
                used_reduced_at_least_once.add(best_key)

                if "conf" in base and "conf" in best_pred:
                    conf_drops.append(base["conf"] - best_pred["conf"])
            else:
                fn += 1

    # FP = reduced detections that never matched any baseline detection
    for frame_id, dets in reduced_dets.items():
        for det_idx, _ in enumerate(dets):
            key = (frame_id, det_idx)
            if key not in used_reduced_at_least_once:
                fp += 1

    return summarize(tp, fp, fn, conf_drops)


def evaluate_sequence(
    seq_name,
    full_yolo_dir,
    reduced_yolo_dir,
    kept_frames_dir,
    output_dir,
    iou_thresh=0.5,
    max_temporal_gap=5,
    match_class=True,
):
    baseline_path = full_yolo_dir / f"{seq_name}_full_yolo_detections.json"
    reduced_path = reduced_yolo_dir / f"{seq_name}_reduced_yolo_detections.json"
    kept_path = kept_frames_dir / f"{seq_name}_kept_frames.json"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline detections: {baseline_path}")

    if not reduced_path.exists():
        raise FileNotFoundError(f"Missing reduced detections: {reduced_path}")

    if not kept_path.exists():
        raise FileNotFoundError(f"Missing kept frames: {kept_path}")

    baseline = normalize_frame_dict(load_json(baseline_path))
    reduced = normalize_frame_dict(load_json(reduced_path))
    kept_frames = [int(x) for x in load_json(kept_path)]

    num_frames = len(baseline)
    num_kept = len(kept_frames)

    keep_eval = evaluate_keep_frames_only(
        baseline,
        reduced,
        kept_frames,
        iou_thresh=iou_thresh,
        match_class=match_class,
    )

    temporal_eval = evaluate_adaptive_temporal(
        baseline,
        reduced,
        kept_frames,
        iou_thresh=iou_thresh,
        match_class=match_class,
        max_temporal_gap=max_temporal_gap,
    )

    result = {
        "sequence": seq_name,
        "keep_frame_only": keep_eval,
        "adaptive_temporal": temporal_eval,
        "num_frames": num_frames,
        "num_kept_frames": num_kept,
        "yolo_run_rate": num_kept / num_frames if num_frames > 0 else 0.0,
        "frame_reduction": 1.0 - (num_kept / num_frames) if num_frames > 0 else 0.0,
        "iou_thresh": iou_thresh,
        "max_temporal_gap": max_temporal_gap,
        "match_class": match_class,
    }

    output_path = output_dir / f"{seq_name}_reduced_detection_eval.json"
    save_json(result, output_path)

    return result


def aggregate_results(sequence_results):
    def sum_metric(section, key):
        return sum(r[section][key] for r in sequence_results)

    keep_tp = sum_metric("keep_frame_only", "tp")
    keep_fp = sum_metric("keep_frame_only", "fp")
    keep_fn = sum_metric("keep_frame_only", "fn")

    temporal_tp = sum_metric("adaptive_temporal", "tp")
    temporal_fp = sum_metric("adaptive_temporal", "fp")
    temporal_fn = sum_metric("adaptive_temporal", "fn")

    total_frames = sum(r["num_frames"] for r in sequence_results)
    total_kept = sum(r["num_kept_frames"] for r in sequence_results)

    return {
        "keep_frame_only": summarize(keep_tp, keep_fp, keep_fn, []),
        "adaptive_temporal": summarize(temporal_tp, temporal_fp, temporal_fn, []),
        "num_sequences": len(sequence_results),
        "num_frames": total_frames,
        "num_kept_frames": total_kept,
        "yolo_run_rate": total_kept / total_frames if total_frames > 0 else 0.0,
        "frame_reduction": 1.0 - (total_kept / total_frames) if total_frames > 0 else 0.0,
    }


def main():
    full_yolo_dir = Path("outputs/full_yolo_baseline_gt/json")
    reduced_yolo_dir = Path("outputs/adaptive_inference_gt/reduced_yolo_json")
    kept_frames_dir = Path("outputs/adaptive_inference_gt/kept_frames")
    output_dir = Path("outputs/adaptive_inference_gt/reduced_detection_eval")

    iou_thresh = 0.5
    max_temporal_gap = 5
    match_class = True

    sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ]

    sequence_results = []

    for seq_name in sequences:
        print(f"\nEvaluating {seq_name}")

        result = evaluate_sequence(
            seq_name=seq_name,
            full_yolo_dir=full_yolo_dir,
            reduced_yolo_dir=reduced_yolo_dir,
            kept_frames_dir=kept_frames_dir,
            output_dir=output_dir,
            iou_thresh=iou_thresh,
            max_temporal_gap=max_temporal_gap,
            match_class=match_class,
        )

        sequence_results.append(result)

        print(json.dumps(result, indent=2))

    aggregate = aggregate_results(sequence_results)

    save_json(
        {
            "per_sequence": sequence_results,
            "aggregate": aggregate,
        },
        output_dir / "all_sequences_reduced_detection_eval.json",
    )

    print("\nAggregate results:")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()