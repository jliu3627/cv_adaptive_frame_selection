from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


def run_yolo_on_sequence(sequence_dir: Path, output_csv: Path, model) -> None:
    """
    Run YOLO on all frames in a MOT17 sequence and save detections to CSV.

    Args:
        sequence_dir: Path to sequence folder, e.g. MOT17-10-SDP
        output_csv: Path to output CSV file
        model: Loaded YOLO model
    """
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    rows = []

    for img_path in tqdm(image_paths, desc=f"Processing {sequence_dir.name}"):
        frame_id = int(img_path.stem)

        results = model(str(img_path), verbose=False)

        if not results:
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = box
            rows.append(
                {
                    "frame": frame_id,
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved detections to {output_csv}")


def main():
    base_dir = Path("data/raw/MOT17/train")
    output_dir = Path("data/interim/detections")

    sequences = [
        "MOT17-10-SDP",
        "MOT17-11-SDP",
    ]

    # You can change this to yolov8s.pt later if needed
    model = YOLO("yolov8n.pt")

    for seq_name in sequences:
        sequence_dir = base_dir / seq_name
        output_csv = output_dir / f"{seq_name}.csv"
        run_yolo_on_sequence(sequence_dir, output_csv, model)


if __name__ == "__main__":
    main()