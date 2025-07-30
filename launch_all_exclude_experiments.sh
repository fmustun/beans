#!/bin/bash

# Launcher script for exclude class experiments with all model types and variants
# This script submits sbatch jobs for dolph2vec variants, biolingual, and aves models
# Excludes one specific class (SW_Luna) from training/validation for all models
# Uses only linear classifiers

# Define the experiments to run
# Format: "model_type:variant:classifier_type:job_name_suffix"
declare -a experiments=(
    # Dolph2Vec variants with linear classifier only
    "dolph2vec:base:linear:dolph2vec_base"
    "dolph2vec:clean:linear:dolph2vec_clean"
    "dolph2vec:32:linear:dolph2vec_32"
    "dolph2vec:128:linear:dolph2vec_128"
    # Biolingual with linear classifier only
    "biolingual::linear:biolingual"
    # Aves with linear classifier only
    "aves::linear:aves"
)

# Class to exclude (only one class)
EXCLUDE_CLASS="SW_Luna"

# Base directory for outputs
BASE_OUTPUT_DIR="logs/exclude_class_experiments"

# Function to create a temporary script for each experiment
create_experiment_script() {
    local model_type=$1
    local variant=$2
    local classifier_type=$3
    local job_suffix=$4
    local job_name="exclude_${EXCLUDE_CLASS}_${job_suffix}"
    local output_dir="${BASE_OUTPUT_DIR}/${model_type}_${job_suffix}"
    
    # Create temporary script
    local temp_script="temp_exclude_${EXCLUDE_CLASS}_${job_suffix}_script.sh"
    
    # Build the command with appropriate arguments
    local cmd_args="--dataset dolphin_reef --task classification --model-type ${model_type} --exclude-class ${EXCLUDE_CLASS} --classifier-type ${classifier_type} --batch-size 32 --epochs 30 --lrs \"[0.0001, 0.00005, 0.00001]\" --num-workers 4 --seed 42 --log-path ${output_dir}.log"
    
    # Add dolph2vec variant only if it's a dolph2vec model
    if [ "$model_type" = "dolph2vec" ]; then
        cmd_args="${cmd_args} --dolph2vec-variant ${variant}"
    fi
    
    cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/err/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=mustun@bio.ens.psl.eu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=ioc@v100

module purge
module load pytorch-gpu/py3/2.0.0
conda activate beans

cd \$WORK/beans

# Create output directory
mkdir -p "${BASE_OUTPUT_DIR}"

python scripts/evaluate_exclude_class.py ${cmd_args}
EOF

    echo "$temp_script"
}

# Main execution
echo "Launching exclude class experiments for all model types and variants..."
echo "Excluding class: ${EXCLUDE_CLASS}"
echo "Classifier type: linear (only)"
echo "Experiments to run:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r model_type variant classifier_type job_suffix <<< "$exp"
    if [ "$model_type" = "dolph2vec" ]; then
        echo "  - ${model_type} (${variant}) ${classifier_type} -> ${job_suffix}"
    else
        echo "  - ${model_type} ${classifier_type} -> ${job_suffix}"
    fi
done
echo ""

# Create output directories if they don't exist
mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "logs/out"
mkdir -p "logs/err"

# Submit jobs for each experiment
total_jobs=0
for exp in "${experiments[@]}"; do
    IFS=':' read -r model_type variant classifier_type job_suffix <<< "$exp"
    
    echo "Submitting job for: ${model_type}"
    if [ "$model_type" = "dolph2vec" ]; then
        echo "  Variant: ${variant}"
    fi
    echo "  Classifier: ${classifier_type}"
    echo "  Exclude class: ${EXCLUDE_CLASS}"
    
    # Create temporary script for this experiment
    temp_script=$(create_experiment_script "$model_type" "$variant" "$classifier_type" "$job_suffix")
    
    # Submit the job
    job_id=$(sbatch "$temp_script" | awk '{print $4}')
    
    echo "  Job submitted with ID: ${job_id}"
    echo "  Job name: exclude_${EXCLUDE_CLASS}_${job_suffix}"
    echo "  Output directory: ${BASE_OUTPUT_DIR}/${model_type}_${job_suffix}"
    echo ""
    
    # Clean up temporary script
    rm "$temp_script"
    
    total_jobs=$((total_jobs + 1))
done

echo "All jobs submitted successfully!"
echo "Total jobs submitted: ${total_jobs}"
echo "You can monitor job status with: squeue -u \$USER"
echo ""
echo "Expected output structure:"
echo "${BASE_OUTPUT_DIR}/"
echo "├── dolph2vec_base.log"
echo "├── dolph2vec_clean.log"
echo "├── dolph2vec_32.log"
echo "├── dolph2vec_128.log"
echo "├── biolingual.log"
echo "└── aves.log" 