from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


def run_yolo_on_frame(model, image_path: Path, conf_thresh=0.25):
    results = model(str(image_path), verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections = []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    for box, conf, cls_id in zip(xyxy, confs, classes):
        conf = float(conf)
        if conf < conf_thresh:
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


def draw_detections(frame, detections, color=(0, 255, 0)):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["conf"]
        cls_id = det["cls"]

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


def draw_csv_detections(frame, det_rows, color=(0, 255, 0)):
    for _, row in det_rows.iterrows():
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        conf = float(row["confidence"])
        cls_id = int(row["class_id"])

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


def draw_top_banner(frame, text, subtext=None):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)

    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtext:
        cv2.putText(
            frame,
            subtext,
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    return frame


def make_transition_slide(width, height, title, subtitle, num_frames=60):
    slides = []

    for _ in range(num_frames):
        frame = 30 * (cv2.UMat(height, width, cv2.CV_8UC3).get())
        frame[:] = (20, 20, 20)

        cv2.putText(
            frame,
            title,
            (60, height // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            subtitle,
            (60, height // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        slides.append(frame)

    return slides


def make_rf_comparison_video(
    sequence_name,
    sequence_dir,
    decisions_csv,
    detections_csv,
    output_video,
    yolo_model_name="yolov8n.pt",
    yolo_conf_thresh=0.25,
    output_fps=30,
):
    img_dir = sequence_dir / "img1"
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No images found in {img_dir}")

    decisions_df = pd.read_csv(decisions_csv)
    detections_df = pd.read_csv(detections_csv)

    decisions_df["frame"] = decisions_df["frame"].astype(int)
    detections_df["frame"] = detections_df["frame"].astype(int)

    decisions_by_frame = decisions_df.set_index("frame").to_dict(orient="index")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, output_fps, (width, height))

    yolo_model = YOLO(yolo_model_name)

    total_frames = len(image_paths)
    kept_frames = set(decisions_df.loc[decisions_df["yolo_ran"] == 1, "frame"].astype(int))
    yolo_run_rate = len(kept_frames) / total_frames
    frame_reduction = 1.0 - yolo_run_rate

    # Transition 1
    for slide in make_transition_slide(
        width,
        height,
        "Part 1: Full Original Video",
        "YOLO is run on every frame",
        num_frames=60,
    ):
        writer.write(slide)

    # Part 1: full original YOLO
    for idx, image_path in enumerate(tqdm(image_paths, desc="Writing full YOLO section")):
        frame_id = int(image_path.stem)
        frame = cv2.imread(str(image_path))

        detections = run_yolo_on_frame(
            yolo_model,
            image_path,
            conf_thresh=yolo_conf_thresh,
        )

        frame = draw_detections(frame, detections, color=(0, 255, 0))
        frame = draw_top_banner(
            frame,
            "Full-frame YOLO baseline",
            f"Frame {frame_id} | YOLO runs on every frame",
        )

        writer.write(frame)

    # Transition 2
    for slide in make_transition_slide(
        width,
        height,
        "Part 2: Reduced-frame RF Output",
        f"Only kept frames are shown | YOLO run rate: {yolo_run_rate:.1%} | Frame reduction: {frame_reduction:.1%}",
        num_frames=60,
    ):
        writer.write(slide)

    # Part 2: reduced video, kept frames only
    kept_image_paths = [
        p for p in image_paths
        if int(p.stem) in kept_frames
    ]

    for image_path in tqdm(kept_image_paths, desc="Writing reduced RF section"):
        frame_id = int(image_path.stem)
        frame = cv2.imread(str(image_path))

        det_rows = detections_df[
            (detections_df["frame"] == frame_id) &
            (detections_df["source"] == "yolo")
        ]

        decision = decisions_by_frame.get(frame_id, {})
        keep_prob = float(decision.get("keep_probability", 1.0))
        forced_refresh = int(decision.get("forced_refresh", 0))

        frame = draw_csv_detections(frame, det_rows, color=(0, 255, 0))
        frame = draw_top_banner(
            frame,
            "Reduced-frame RF output",
            f"Frame {frame_id} | YOLO ran | keep_prob={keep_prob:.3f} | forced_refresh={forced_refresh}",
        )

        writer.write(frame)

    writer.release()

    print(f"Saved comparison video to: {output_video}")
    print(f"Total frames: {total_frames}")
    print(f"Kept frames: {len(kept_frames)}")
    print(f"YOLO run rate: {yolo_run_rate:.3f}")
    print(f"Frame reduction: {frame_reduction:.3f}")


def main():
    sequence_name = "MOT17-03-FRCNN"

    sequence_dir = Path("data/raw/MOT17/test") / sequence_name

    decisions_csv = Path(
        f"outputs/test/adaptive_inference_gt/decisions/{sequence_name}_decisions.csv"
    )

    detections_csv = Path(
        f"outputs/test/adaptive_inference_gt/detections/{sequence_name}_adaptive_detections.csv"
    )

    output_video = Path(
        f"outputs/videos/{sequence_name}_rf_comparison_video.mp4"
    )

    make_rf_comparison_video(
        sequence_name=sequence_name,
        sequence_dir=sequence_dir,
        decisions_csv=decisions_csv,
        detections_csv=detections_csv,
        output_video=output_video,
        yolo_model_name="yolov8n.pt",
        yolo_conf_thresh=0.25,
        output_fps=30,
    )


if __name__ == "__main__":
    main()