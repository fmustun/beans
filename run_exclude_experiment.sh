#!/bin/bash

# Script to run a single exclude class experiment
# Usage: ./scripts/run_single_exclude_experiment.sh <model_type> <exclude_class> <classifier_type> [dolph2vec_variant]

set -e  # Exit on any error

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <model_type> <exclude_class> <classifier_type> [dolph2vec_variant]"
    echo "Example: $0 aves SW_Yosefa mlp"
    echo "Example: $0 dolph2vec SW_Yosefa mlp base"
    exit 1
fi

MODEL_TYPE=$1
EXCLUDE_CLASS=$2
CLASSIFIER_TYPE=$3
DOLPH2VEC_VARIANT=$4

# Configuration
DATASET="dolphin_reef"
TASK="classification"
BATCH_SIZE=32
EPOCHS=30
LRS="[0.0001, 0.00005, 0.00001]"
SEED=42
NUM_WORKERS=4

# Create logs directory
mkdir -p logs/exclude_class_experiments

# Build log filename
LOG_FILE="logs/exclude_class_experiments/${MODEL_TYPE}_${EXCLUDE_CLASS}_${CLASSIFIER_TYPE}"
if [ "$MODEL_TYPE" = "dolph2vec" ]; then
    if [ -z "$DOLPH2VEC_VARIANT" ]; then
        echo "Error: dolph2vec_variant must be specified for dolph2vec model type"
        echo "Available variants: base, 32, 128, clean"
        exit 1
    fi
    LOG_FILE="${LOG_FILE}_${DOLPH2VEC_VARIANT}"
fi
LOG_FILE="${LOG_FILE}.log"

echo "Running experiment:"
echo "  Model: $MODEL_TYPE"
echo "  Exclude class: $EXCLUDE_CLASS"
echo "  Classifier: $CLASSIFIER_TYPE"
if [ "$MODEL_TYPE" = "dolph2vec" ]; then
    echo "  Dolph2Vec variant: $DOLPH2VEC_VARIANT"
fi
echo "  Log file: $LOG_FILE"
echo "----------------------------------------"

# Build command
CMD="python scripts/evaluate_exclude_class.py \
    --dataset $DATASET \
    --task $TASK \
    --model-type $MODEL_TYPE \
    --exclude-class $EXCLUDE_CLASS \
    --classifier-type $CLASSIFIER_TYPE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lrs '$LRS' \
    --num-workers $NUM_WORKERS \
    --seed $SEED \
    --log-path $LOG_FILE"

# Add dolph2vec variant if specified
if [ "$MODEL_TYPE" = "dolph2vec" ] && [ -n "$DOLPH2VEC_VARIANT" ]; then
    CMD="$CMD --dolph2vec-variant $DOLPH2VEC_VARIANT"
fi

# Run the experiment
echo "Executing: $CMD"
eval $CMD

echo "Experiment completed!"
echo "Results saved in: $LOG_FILE"
