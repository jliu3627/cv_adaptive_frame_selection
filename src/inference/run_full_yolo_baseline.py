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


def run_full_yolo_for_sequence(
    sequence_dir: Path,
    yolo_model,
    output_json: Path,
    yolo_conf_thresh=0.25,
):
    img_dir = sequence_dir / "img1"
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No images found in {img_dir}")

    full_detections = {}

    for image_path in tqdm(image_paths, desc=f"Full YOLO {sequence_dir.name}"):
        frame_id = int(image_path.stem)

        detections = run_yolo_on_frame(
            yolo_model,
            image_path,
            confidence_thresh=yolo_conf_thresh,
        )

        full_detections[frame_id] = detections

    save_json(full_detections, output_json)
    print(f"Saved full YOLO detections to {output_json}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    output_dir = Path("outputs/full_yolo_baseline_gt/json")

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

    yolo_model = YOLO("yolov8n.pt")

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name
        output_json = output_dir / f"{seq_name}_full_yolo_detections.json"

        run_full_yolo_for_sequence(
            sequence_dir=sequence_dir,
            yolo_model=yolo_model,
            output_json=output_json,
            yolo_conf_thresh=0.25,
        )


if __name__ == "__main__":
    main()