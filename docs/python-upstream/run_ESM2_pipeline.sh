#!/bin/bash

#$ -cwd
#$ -l h_rt=10:00:00          # Adjust runtime as needed
#$ -N ESM2_Pipeline          # Job name
#$ -pe sharedmem 16          # Number of CPU cores
#$ -l h_vmem=32G             # Memory per core
#$ -l rl9=false
#$ -m beas                   # Send email on job status
#$ -M s2488128@ed.ac.uk # Your email for notifications

# Log files for stdout and stderr
exec > /exports/eddie/scratch/s2488128/ESM2/logs/Pipeline_800_t12_35m.out \
     2>&1  # Redirect both stdout and stderr to the .out file

# Load Python 3.11
echo "Loading Python version 3.11.4..."
module load python/3.11.4
export LD_LIBRARY_PATH=/exports/applications/apps/SL7/python/3.11.3/lib:$LD_LIBRARY_PATH
echo "Python loaded successfully."

# Set Hugging Face and PyTorch cache directories to scratch space
export HF_HOME=/exports/eddie/scratch/s2488128/hf_cache
export TORCH_HOME=/exports/eddie/scratch/s2488128/torch_cache
echo "Cache directories set to scratch space."

# Activate the virtual environment
echo "Activating virtual environment..."
source /exports/eddie/scratch/s2488128/venv/bin/activate
echo "Virtual environment activated successfully."

# Function to run a task and check for errors
run_task() {
    local script_name=$1
    local task_name=$2

    echo "$(date) - Starting ${task_name}..."
    python $script_name

    if [ $? -eq 0 ]; then
        echo "$(date) - ${task_name} completed successfully."
    else
        echo "$(date) - Error encountered during ${task_name}. Exiting pipeline."
        exit 1
    fi
}

# Run tasks in sequence
run_task "/exports/eddie/scratch/s2488128/ESM2/ESM2Embeddings.py" "ESM2 Embeddings"
run_task "/exports/eddie/scratch/s2488128/ESM2/ESM2Stacking.py" "Stacking .pt Files"
run_task "/exports/eddie/scratch/s2488128/ESM2/ESM2SAttentionWeighter.py" "Weighted Embeddings Computation"
run_task "/exports/eddie/scratch/s2488128/ESM2/ESM2AttentionOptimiser.py" "Attention Optimisation"

# Print final completion message
echo "$(date) - All tasks in ESM2 pipeline completed successfully."