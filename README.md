# Adaptive Frame Selection for Object Detection

This project uses a Random Forest adaptive frame-selection model to reduce how often YOLO is run on video frames while preserving object detection quality.

## Included Model and Metrics

Latest trained model:

```text
outputs/models/random_forest/random_forest_gt_model.joblib
```

Saved model metrics:

```text
outputs/models/random_forest/metrics.json
outputs/models/random_forest/feature_importance.csv
```

This is the latest model from the main project pipeline. It was trained using all available training sequences in the local project workflow.

The Docker demo uses a packaged demo model here:

```text
outputs/models/random_forest_all_variants/random_forest_gt_model.joblib
```

That Docker model is included so the demo can run directly on the bundled MOT17-02 DPM, FRCNN, and SDP sequences without retraining.

## One-Command Demo

From the project root, run:

```bash
./run_docker.sh
```

This builds the Docker image and runs the latest model on the included MOT17-02 DPM, FRCNN, and SDP demo sequences.

The script runs:

```bash
python run_latest_model.py --sequence-ids 02 --variants DPM FRCNN SDP
```

Expected generated outputs include:

```text
outputs/full_yolo_baseline_gt/json/
outputs/adaptive_inference_gt/
outputs/adaptive_inference_gt/reduced_detection_eval/
outputs/evaluation/adaptive_vs_gt_all_variants/
```

## Manual Docker Commands

On Apple Silicon:

```bash
docker build --platform linux/arm64 -t adaptive-frame-selection .
docker run --rm -it adaptive-frame-selection
```

On x86:

```bash
docker build -t adaptive-frame-selection .
docker run --rm -it adaptive-frame-selection
```

The container entrypoint is:

```bash
bash scripts/run_project.sh
```

## Submission Notes

The video demonstration shows the Docker command being typed in the terminal and running successfully. It also shows the model inference/evaluation flow completing with the included latest model.

## Optional Local Pipeline

If you want to rerun the full local pipeline outside Docker, use:

```bash
python src/data/read_mot17.py
python src/labeling/generate_labels.py
python src/features/extract_cv_features.py
python src/models/train_rf.py
python src/inference/run_full_yolo_baseline.py
python src/inference/run_rf_adaptive_model_yolo.py
python src/evaluation/evaluate_reduced_detection.py
```

Optional k-frame baseline:

```bash
python src/inference/run_kframe_yolo.py
python src/evaluation/evaluate_kframe_detection.py
```
