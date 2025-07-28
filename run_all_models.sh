#!/bin/bash

# Script to run all models on dolphin_reef dataset with hardcoded parameters
# Creates a timestamped folder in logs/ and puts all model logs in it

# Get timestamp for the run folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create logs directory and timestamped subdirectory
RUN_DIR="logs/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "🚀 Starting all model runs on dolphin_reef dataset"
echo "📁 Run directory: $RUN_DIR"
echo "⏰ Timestamp: $TIMESTAMP"
echo ""

# Common parameters (hardcoded)
DATASET="dolphin_reef"
TASK="classification"
BATCH_SIZE=16
EPOCHS=50
LRS="[0.0001, 0.00005, 0.00001]"
CLASSIFIER_TYPE="linear"
NUM_WORKERS=4
SEED=42

echo "🔧 Common parameters:"
echo "   Dataset: $DATASET"
echo "   Task: $TASK"
echo "   Batch size: $BATCH_SIZE"
echo "   Epochs: $EPOCHS"
echo "   Learning rates: $LRS"
echo "   Classifier type: $CLASSIFIER_TYPE"
echo "   Num workers: $NUM_WORKERS"
echo "   Seed: $SEED"
echo ""

# Function to run a model
run_model() {
    local model_type=$1
    local variant=$2
    local log_file="${RUN_DIR}/${model_type}"
    
    if [ ! -z "$variant" ]; then
        log_file="${log_file}_${variant}"
    fi
    log_file="${log_file}.log"
    
    echo "🎯 Running $model_type"
    if [ ! -z "$variant" ]; then
        echo "   Variant: $variant"
    fi
    echo "   Log file: $log_file"
    echo ""
    
    # Build command
    cmd="python scripts/evaluate.py"
    cmd="$cmd --dataset $DATASET"
    cmd="$cmd --model-type $model_type"
    cmd="$cmd --task $TASK"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --lrs '$LRS'"
    cmd="$cmd --classifier-type $CLASSIFIER_TYPE"
    cmd="$cmd --num-workers $NUM_WORKERS"
    cmd="$cmd --seed $SEED"
    cmd="$cmd --log-path $log_file"
    
    # Add variant for Dolph2Vec
    if [ "$model_type" = "dolph2vec" ] && [ ! -z "$variant" ]; then
        cmd="$cmd --dolph2vec-variant $variant"
    fi
    
    echo "Running: $cmd"
    echo "=========================================="
    
    # Execute command
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_type completed successfully"
    else
        echo "❌ $model_type failed"
    fi
    echo ""
}

# Run AVES
run_model "aves"

# Run Biolingual
run_model "biolingual"

# Run Dolph2Vec variants
run_model "dolph2vec" "base"
run_model "dolph2vec" "32"
run_model "dolph2vec" "128"
run_model "dolph2vec" "clean"

echo "🎉 All model runs completed!"
echo "📁 All logs saved in: $RUN_DIR"
echo "📋 Log files:"
ls -la "$RUN_DIR"/*.log 