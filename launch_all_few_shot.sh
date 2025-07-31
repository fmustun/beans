#!/bin/bash

# Launcher script for few-shot experiments with all model types and variants
# This script submits sbatch jobs for dolph2vec variants, biolingual, and aves models

# Define the experiments to run
# Format: "model_type:variant:job_name_suffix"
declare -a experiments=(
    "dolph2vec:base:base"
    "dolph2vec:clean:clean"
    "dolph2vec:32:32"
    "dolph2vec:128:128"
    "biolingual::biolingual"
    #"aves::aves"
)

# Base directory for outputs
BASE_OUTPUT_DIR="logs/few_shot"

# Function to create a temporary script for each experiment
create_experiment_script() {
    local model_type=$1
    local variant=$2
    local job_suffix=$3
    local job_name="few_shot_${job_suffix}"
    local output_dir="${BASE_OUTPUT_DIR}/${model_type}_${job_suffix}"
    
    # Create temporary script
    local temp_script="temp_${job_suffix}_script.sh"
    
    # Build the command with appropriate arguments
    local cmd_args="--samples-per-class 50 100 200 --model-type ${model_type} --classifier-type linear --task classification --output-dir ${output_dir} --batch-size 32 --lrs \"[0.0001, 0.00005, 0.00001]\" --epochs 30"
    
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
#SBATCH --time=08:00:00
#SBATCH --mail-user=mustun@bio.ens.psl.eu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=ioc@v100

module purge
module load pytorch-gpu/py3/2.0.0
conda activate beans

cd \$WORK/beans

python scripts/few_shot_dolphin_reef.py ${cmd_args}
EOF

    echo "$temp_script"
}

# Main execution
echo "Launching few-shot experiments for all model types and variants..."
echo "Experiments to run:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r model_type variant job_suffix <<< "$exp"
    if [ "$model_type" = "dolph2vec" ]; then
        echo "  - ${model_type} (${variant}) -> ${job_suffix}"
    else
        echo "  - ${model_type} -> ${job_suffix}"
    fi
done
echo ""

# Create output directories if they don't exist
mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "logs/out"
mkdir -p "logs/err"

# Submit jobs for each experiment
for exp in "${experiments[@]}"; do
    IFS=':' read -r model_type variant job_suffix <<< "$exp"
    
    echo "Submitting job for: ${model_type}"
    if [ "$model_type" = "dolph2vec" ]; then
        echo "  Variant: ${variant}"
    fi
    
    # Create temporary script for this experiment
    temp_script=$(create_experiment_script "$model_type" "$variant" "$job_suffix")
    
    # Submit the job
    job_id=$(sbatch "$temp_script" | awk '{print $4}')
    
    echo "  Job submitted with ID: ${job_id}"
    echo "  Job name: few_shot_${job_suffix}"
    echo "  Output directory: ${BASE_OUTPUT_DIR}/${model_type}_${job_suffix}"
    echo ""
    
    # Clean up temporary script
    rm "$temp_script"
done

echo "All jobs submitted successfully!"
echo "You can monitor job status with: squeue -u \$USER" 