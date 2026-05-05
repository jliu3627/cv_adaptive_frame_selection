#!/usr/bin/env bash

set -e

echo "============================================================"
echo "Adaptive Frame Selection Final Project Runner"
echo "============================================================"

echo ""
echo "Python version:"
python --version

echo ""
echo "Checking demo dataset folders..."

for variant in DPM FRCNN SDP; do
    if [ ! -d "data/raw/MOT17/train/MOT17-02-${variant}" ]; then
        echo "ERROR: Missing demo dataset folder:"
        echo "  data/raw/MOT17/train/MOT17-02-${variant}"
        exit 1
    fi
done

echo ""
echo "Checking latest trained model..."

if [ ! -f "outputs/models/random_forest_all_variants/random_forest_gt_model.joblib" ]; then
    echo "ERROR: Missing latest trained model:"
    echo "  outputs/models/random_forest_all_variants/random_forest_gt_model.joblib"
    exit 1
fi

echo ""
echo "Running latest trained model on MOT17-02-DPM, MOT17-02-FRCNN, and MOT17-02-SDP without retraining..."

python run_latest_model.py \
    --sequence-ids 02 \
    --variants DPM FRCNN SDP

echo ""
echo "Project completed successfully."