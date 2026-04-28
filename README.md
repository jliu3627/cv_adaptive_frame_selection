# Run scripts:

### Read data:
```
python src/data/read_mot17.py
```

### Generate GT labels:
```
python src/labeling/generate_labels.py
```

### Compute CV features:
```
python src/features/extract_cv_features.py
```

### Train RF model:
```
python src/models/train_rf.py
```

### Inference:
```
python src/inference/run_full_yolo_baseline.py
python src/inference/run_rf_adaptive_model_yolo.py
python src/inference/run_kframe_yolo.py # optional
```

### Evaluation
```
python src/evaluation/evaluate_reduced_detection.py
python src/evaluation/evaluate_kframe_detection.py # optional
```