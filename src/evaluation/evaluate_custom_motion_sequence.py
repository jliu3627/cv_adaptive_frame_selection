from pathlib import Path
import json
import numpy as np
import pandas as pd


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


def summarize_detection(tp, fp, fn, conf_drops):
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


def evaluate_keep_frames_only(baseline, reduced, kept_frames, iou_thresh=0.5, match_class=True):
    tp = fp = fn = 0
    conf_drops = []

    for frame_id in sorted(kept_frames):
        base = baseline.get(frame_id, [])
        pred = reduced.get(frame_id, [])

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

    return summarize_detection(tp, fp, fn, conf_drops)


def nearest_kept_frames(frame_id, kept_frames_sorted):
    kept = np.array(kept_frames_sorted)
    idx = np.searchsorted(kept, frame_id)

    prev_frame = int(kept[idx - 1]) if idx > 0 else None
    next_frame = int(kept[idx]) if idx < len(kept) else None

    return prev_frame, next_frame


def evaluate_adaptive_temporal(
    baseline,
    reduced,
    kept_frames,
    iou_thresh=0.5,
    match_class=True,
    max_temporal_gap=5,
):
    kept_frames = set(kept_frames)
    kept_frames_sorted = sorted(kept_frames)

    tp = fp = fn = 0
    conf_drops = []
    used_reduced_at_least_once = set()

    for frame_id in sorted(baseline.keys()):
        base_dets = baseline.get(frame_id, [])
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
            for det_idx, det in enumerate(reduced.get(candidate_frame, [])):
                candidate_preds.append(
                    {
                        **det,
                        "_source_frame": candidate_frame,
                        "_source_idx": det_idx,
                    }
                )

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

    for frame_id, dets in reduced.items():
        for det_idx, _ in enumerate(dets):
            key = (frame_id, det_idx)
            if key not in used_reduced_at_least_once:
                fp += 1

    return summarize_detection(tp, fp, fn, conf_drops)


def summarize_decisions(df):
    num_frames = len(df)
    yolo_runs = int(df["yolo_ran"].sum())
    predicted_keeps = int(df["pred_label"].sum())
    forced_refreshes = int(df["forced_refresh"].sum())
    expected_keeps = int(df["expected_label"].sum())
    expected_skips = num_frames - expected_keeps

    false_keeps = int(((df["expected_label"] == 0) & (df["pred_label"] == 1)).sum())
    false_skips = int(((df["expected_label"] == 1) & (df["pred_label"] == 0)).sum())
    correct = int((df["expected_label"] == df["pred_label"]).sum())

    return {
        "num_frames": num_frames,
        "expected_keeps": expected_keeps,
        "expected_skips": expected_skips,
        "predicted_keeps": predicted_keeps,
        "predicted_skips": num_frames - predicted_keeps,
        "yolo_runs": yolo_runs,
        "forced_refreshes": forced_refreshes,
        "yolo_run_rate": yolo_runs / num_frames if num_frames > 0 else 0.0,
        "frame_reduction": 1.0 - (yolo_runs / num_frames) if num_frames > 0 else 0.0,
        "decision_accuracy_vs_expected_phase": correct / num_frames if num_frames > 0 else 0.0,
        "false_keeps": false_keeps,
        "false_skips": false_skips,
        "avg_keep_probability": float(df["keep_probability"].mean()) if num_frames > 0 else None,
    }


def load_custom_decision_frame(manifest_path, decisions_path):
    manifest_df = pd.read_csv(manifest_path)
    decisions_df = pd.read_csv(decisions_path)

    merged = decisions_df.merge(
        manifest_df,
        left_on="frame",
        right_on="generated_frame",
        how="left",
        validate="one_to_one",
    )

    if merged["phase"].isna().any():
        missing = merged.loc[merged["phase"].isna(), "frame"].tolist()
        raise ValueError(f"Missing manifest rows for generated frames: {missing[:10]}")

    # Expected behavior for this synthetic stress test:
    # low_repeated frames should mostly SKIP; high_random frames should mostly KEEP.
    merged["expected_label"] = (merged["phase"] == "high_random").astype(int)
    return merged


def evaluate_phase_behavior(merged_df):
    phase_results = {}
    for phase, phase_df in merged_df.groupby("phase"):
        phase_results[phase] = summarize_decisions(phase_df)

    return {
        "overall": summarize_decisions(merged_df),
        "by_phase": phase_results,
    }


def evaluate_detection_if_available(
    seq_name,
    full_yolo_dir,
    reduced_yolo_dir,
    kept_frames_dir,
    iou_thresh=0.5,
    max_temporal_gap=5,
    match_class=True,
):
    baseline_path = full_yolo_dir / f"{seq_name}_full_yolo_detections.json"
    reduced_path = reduced_yolo_dir / f"{seq_name}_reduced_yolo_detections.json"
    kept_path = kept_frames_dir / f"{seq_name}_kept_frames.json"

    if not baseline_path.exists():
        return {
            "available": False,
            "reason": f"Missing baseline detections: {baseline_path}",
        }

    baseline = normalize_frame_dict(load_json(baseline_path))
    reduced = normalize_frame_dict(load_json(reduced_path))
    kept_frames = [int(x) for x in load_json(kept_path)]

    num_frames = len(baseline)
    num_kept = len(kept_frames)

    return {
        "available": True,
        "keep_frame_only": evaluate_keep_frames_only(
            baseline,
            reduced,
            kept_frames,
            iou_thresh=iou_thresh,
            match_class=match_class,
        ),
        "adaptive_temporal": evaluate_adaptive_temporal(
            baseline,
            reduced,
            kept_frames,
            iou_thresh=iou_thresh,
            match_class=match_class,
            max_temporal_gap=max_temporal_gap,
        ),
        "num_frames": num_frames,
        "num_kept_frames": num_kept,
        "yolo_run_rate": num_kept / num_frames if num_frames > 0 else 0.0,
        "frame_reduction": 1.0 - (num_kept / num_frames) if num_frames > 0 else 0.0,
        "iou_thresh": iou_thresh,
        "max_temporal_gap": max_temporal_gap,
        "match_class": match_class,
    }


def main():
    seq_name = "MOT17-custom-lowhigh"
    custom_sequence_dir = Path("data/processed/custom_sequences") / seq_name
    manifest_path = custom_sequence_dir / "frame_manifest.csv"

    adaptive_dir = Path("outputs/custom/adaptive_inference")
    decisions_path = adaptive_dir / "decisions" / f"{seq_name}_decisions.csv"
    reduced_yolo_dir = adaptive_dir / "reduced_yolo_json"
    kept_frames_dir = adaptive_dir / "kept_frames"

    full_yolo_dir = Path("outputs/custom/full_yolo_baseline/json")
    output_dir = adaptive_dir / "evaluation"

    iou_thresh = 0.5
    max_temporal_gap = 5
    match_class = True

    merged_df = load_custom_decision_frame(manifest_path, decisions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_dir / f"{seq_name}_per_frame_eval.csv", index=False)

    result = {
        "sequence": seq_name,
        "phase_behavior": evaluate_phase_behavior(merged_df),
        "detection_eval": evaluate_detection_if_available(
            seq_name=seq_name,
            full_yolo_dir=full_yolo_dir,
            reduced_yolo_dir=reduced_yolo_dir,
            kept_frames_dir=kept_frames_dir,
            iou_thresh=iou_thresh,
            max_temporal_gap=max_temporal_gap,
            match_class=match_class,
        ),
    }

    output_path = output_dir / f"{seq_name}_custom_eval.json"
    save_json(result, output_path)

    print(f"Saved per-frame custom evaluation to {output_dir / f'{seq_name}_per_frame_eval.csv'}")
    print(f"Saved custom evaluation to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
