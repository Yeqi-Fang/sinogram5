#!/bin/bash

# Script to run 6-fold cross-validation for sinogram restoration

# Base directory for data
DATA_DIR="/mnt/d/fyq/sinogram/2e9div_smooth"

# Base directory for outputs
OUTPUT_BASE="cv_results"
MODELS_DIR="${OUTPUT_BASE}/models"
LOG_DIR="${OUTPUT_BASE}/logs"
PREDICTIONS_DIR="${OUTPUT_BASE}/predictions"
FINAL_PREDICTIONS_DIR="${OUTPUT_BASE}/final_predictions"

# Create directories
mkdir -p $MODELS_DIR $LOG_DIR $PREDICTIONS_DIR $FINAL_PREDICTIONS_DIR

# Training parameters
BATCH_SIZE=24
NUM_EPOCHS=30
LEARNING_RATE=5e-6
ATTENTION=1

# Number of folds
NUM_FOLDS=6

# Run all folds
echo "Starting 6-fold cross-validation..."
python main2.py \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --models_dir $MODELS_DIR \
    --log_dir $LOG_DIR \
    --predictions_dir $PREDICTIONS_DIR \
    --attention $ATTENTION \
    --lr $LEARNING_RATE \
    --num_folds $NUM_FOLDS \
    --start_fold 0 \
    --end_fold $(($NUM_FOLDS-1)) \
    --results_log "${OUTPUT_BASE}/cv_results.csv"

# Merge predictions from all folds
echo "Merging predictions from all folds..."
python convert_incomplete_to_predicted.py \
    --predictions_dir $PREDICTIONS_DIR \
    --output_dir $FINAL_PREDICTIONS_DIR

echo "Cross-validation complete!"
echo "Results are saved in ${OUTPUT_BASE}"
echo "Final predictions are in ${FINAL_PREDICTIONS_DIR}"
