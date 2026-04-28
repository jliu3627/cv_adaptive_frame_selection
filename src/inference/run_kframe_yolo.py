from pathlib import Path
import json
from ultralytics import YOLO
from tqdm import tqdm


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


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_kframe_yolo_for_sequence(
    sequence_dir: Path,
    yolo_model,
    output_json: Path,
    kept_frames_output: Path,
    k=5,
    yolo_conf_thresh=0.25,
    always_keep_first=True,
):
    img_dir = sequence_dir / "img1"
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No images found in {img_dir}")

    reduced_detections = {}
    kept_frames = []

    for idx, image_path in enumerate(
        tqdm(image_paths, desc=f"K-frame YOLO {sequence_dir.name}")
    ):
        frame_id = int(image_path.stem)

        keep_frame = (idx % k == 0)

        if always_keep_first and idx == 0:
            keep_frame = True

        if keep_frame:
            detections = run_yolo_on_frame(
                yolo_model,
                image_path,
                confidence_thresh=yolo_conf_thresh,
            )

            reduced_detections[frame_id] = detections
            kept_frames.append(frame_id)
        else:
            reduced_detections[frame_id] = []

    save_json(reduced_detections, output_json)
    save_json(kept_frames, kept_frames_output)

    print(f"Saved k-frame detections to {output_json}")
    print(f"Saved k-frame kept frames to {kept_frames_output}")


def main():
    base_dir = Path("data/raw/MOT17/train")

    output_dir = Path("outputs/kframe_yolo_gt")
    detections_dir = output_dir / "reduced_yolo_json"
    kept_frames_dir = output_dir / "kept_frames"

    k = 5
    yolo_conf_thresh = 0.25

    sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ]

    yolo_model = YOLO("yolov8n.pt")

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name

        output_json = detections_dir / f"{seq_name}_k{k}_reduced_yolo_detections.json"
        kept_frames_output = kept_frames_dir / f"{seq_name}_k{k}_kept_frames.json"

        run_kframe_yolo_for_sequence(
            sequence_dir=sequence_dir,
            yolo_model=yolo_model,
            output_json=output_json,
            kept_frames_output=kept_frames_output,
            k=k,
            yolo_conf_thresh=yolo_conf_thresh,
            always_keep_first=True,
        )

    # test dataset
    base_dir = Path("data/raw/MOT17/test")
    output_dir = Path("outputs/test/kframe_yolo")

    detections_dir = output_dir / "reduced_yolo_json"
    kept_frames_dir = output_dir / "kept_frames"

    sequences = sorted([p.name for p in base_dir.iterdir() if p.is_dir() and "FRCNN" in p.name])

    yolo_model = YOLO("yolov8n.pt")

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name
        img_dir = sequence_dir / "img1"
        image_paths = sorted(img_dir.glob("*.jpg"))

        reduced_detections = {}
        kept_frames = []

        for idx, image_path in enumerate(tqdm(image_paths, desc=f"K-frame test {seq_name}")):
            frame_id = int(image_path.stem)

            keep_frame = idx % k == 0

            if keep_frame:
                detections = run_yolo_on_frame(
                    yolo_model,
                    image_path,
                    confidence_thresh=yolo_conf_thresh,
                )
                reduced_detections[frame_id] = detections
                kept_frames.append(frame_id)
            else:
                reduced_detections[frame_id] = []

        save_json(
            reduced_detections,
            detections_dir / f"{seq_name}_k{k}_reduced_yolo_detections.json",
        )
        save_json(
            kept_frames,
            kept_frames_dir / f"{seq_name}_k{k}_kept_frames.json",
        )

if __name__ == "__main__":
    main()