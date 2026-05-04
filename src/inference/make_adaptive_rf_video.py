from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


FEATURE_COLS = [
    "frame_diff_mean",
    "frame_diff_max",
    "flow_mean",
    "flow_max",
    "flow_std",
    "edge_diff_mean",
    "edge_density_prev",
    "edge_density_curr",
    "hist_bhattacharyya",
    "hist_corr",
]


def load_grayscale(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_features(prev_img_path: Path, curr_img_path: Path):
    prev_gray = load_grayscale(prev_img_path)
    curr_gray = load_grayscale(curr_img_path)

    diff = cv2.absdiff(prev_gray, curr_gray)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    prev_edges = cv2.Canny(prev_gray, 100, 200)
    curr_edges = cv2.Canny(curr_gray, 100, 200)
    edge_diff = cv2.absdiff(prev_edges, curr_edges)

    prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])

    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()

    return {
        "frame_diff_mean": float(np.mean(diff)),
        "frame_diff_max": float(np.max(diff)),
        "flow_mean": float(np.mean(mag)),
        "flow_max": float(np.max(mag)),
        "flow_std": float(np.std(mag)),
        "edge_diff_mean": float(np.mean(edge_diff) / 255.0),
        "edge_density_prev": float(np.mean(prev_edges) / 255.0),
        "edge_density_curr": float(np.mean(curr_edges) / 255.0),
        "hist_bhattacharyya": float(
            cv2.compareHist(
                prev_hist.astype(np.float32),
                curr_hist.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        ),
        "hist_corr": float(
            cv2.compareHist(
                prev_hist.astype(np.float32),
                curr_hist.astype(np.float32),
                cv2.HISTCMP_CORREL,
            )
        ),
    }


def run_yolo_on_frame(model, image_path: Path, confidence_thresh=0.25):
    results = model(str(image_path), verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    detections = []

    for box, conf, cls_id in zip(xyxy, confs, classes):
        conf = float(conf)

        if conf < confidence_thresh:
            continue

        x1, y1, x2, y2 = box

        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": conf,
                "cls": int(cls_id),
            }
        )

    return detections


def draw_detections(frame, detections, source):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["conf"]
        cls_id = det["cls"]

        color = (0, 255, 0) if source == "YOLO" else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"cls {cls_id} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_info_panel(
    frame,
    frame_id,
    idx,
    total_frames,
    yolo_ran,
    keep_prob,
    skip_streak,
    num_kept,
):
    h, w = frame.shape[:2]

    status = "YOLO RUN" if yolo_ran else "SKIPPED / REUSED"
    status_color = (0, 255, 0) if yolo_ran else (0, 165, 255)

    panel_h = 105
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    lines = [
        f"Frame ID: {frame_id}    Index: {idx + 1}/{total_frames}",
        f"Decision: {status}    Keep Probability: {keep_prob:.3f}",
        f"Skip Streak: {skip_streak}    Kept Frames: {num_kept}",
    ]

    y = 25
    for i, text in enumerate(lines):
        color = status_color if i == 1 else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 32

    return frame


def make_adaptive_rf_video(
    sequence_dir: Path,
    rf_model_path: Path,
    output_video_path: Path,
    yolo_model_name="yolov8n.pt",
    keep_threshold=0.45,
    yolo_conf_thresh=0.25,
    max_skip_streak=5,
    output_fps=30,
):
    img_dir = sequence_dir / "img1"
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No images found in {img_dir}")

    rf_model = joblib.load(rf_model_path)
    yolo_model = YOLO(yolo_model_name)

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        output_fps,
        (width, height),
    )

    prev_img_path = None
    last_detections = []
    skip_streak = 0
    num_kept = 0

    for idx, curr_img_path in enumerate(
        tqdm(image_paths, desc=f"Making video {sequence_dir.name}")
    ):
        frame_id = int(curr_img_path.stem)
        frame = cv2.imread(str(curr_img_path))

        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {curr_img_path}")

        if idx == 0:
            detections = run_yolo_on_frame(
                yolo_model,
                curr_img_path,
                confidence_thresh=yolo_conf_thresh,
            )

            last_detections = detections
            yolo_ran = 1
            keep_prob = 1.0
            skip_streak = 0
            num_kept += 1
            source = "YOLO"

        else:
            features = compute_features(prev_img_path, curr_img_path)
            X_curr = pd.DataFrame([features])[FEATURE_COLS]

            keep_prob = float(rf_model.predict_proba(X_curr)[0][1])
            keep_frame = keep_prob >= keep_threshold

            if skip_streak >= max_skip_streak:
                keep_frame = True

            if keep_frame:
                detections = run_yolo_on_frame(
                    yolo_model,
                    curr_img_path,
                    confidence_thresh=yolo_conf_thresh,
                )

                last_detections = detections
                yolo_ran = 1
                skip_streak = 0
                num_kept += 1
                source = "YOLO"

            else:
                detections = last_detections
                yolo_ran = 0
                skip_streak += 1
                source = "REUSED"

        frame = draw_detections(frame, detections, source)
        frame = draw_info_panel(
            frame=frame,
            frame_id=frame_id,
            idx=idx,
            total_frames=len(image_paths),
            yolo_ran=yolo_ran,
            keep_prob=keep_prob,
            skip_streak=skip_streak,
            num_kept=num_kept,
        )

        writer.write(frame)
        prev_img_path = curr_img_path

    writer.release()

    yolo_run_rate = num_kept / len(image_paths)
    frame_reduction = 1.0 - yolo_run_rate

    print(f"Saved video to: {output_video_path}")
    print(f"YOLO run rate: {yolo_run_rate:.3f}")
    print(f"Frame reduction: {frame_reduction:.3f}")


def main():
    sequence_name = "MOT17-03-FRCNN"

    sequence_dir = Path("data/raw/MOT17/test") / sequence_name
    rf_model_path = Path("outputs/models/random_forest/random_forest_gt_model.joblib")

    output_video_path = Path(
        f"outputs/videos/{sequence_name}_adaptive_rf_yolo.mp4"
    )

    make_adaptive_rf_video(
        sequence_dir=sequence_dir,
        rf_model_path=rf_model_path,
        output_video_path=output_video_path,
        yolo_model_name="yolov8n.pt",
        keep_threshold=0.45,
        yolo_conf_thresh=0.25,
        max_skip_streak=5,
        output_fps=30,
    )


if __name__ == "__main__":
    main()