#!/bin/bash
#SBATCH --job-name=imptask          # Job name
#SBATCH --partition=batch           # Partition to use
#SBATCH --nodelist=gpu04      # Specify gpu02
#SBATCH --gres=gpu:4                # Request 4 GPUs per node
#SBATCH --ntasks-per-node=1         # One task per node (torchrun manages GPUs)
#SBATCH --cpus-per-task=40          # Request all CPUs on each node (48 CPUs per node)
#SBATCH --mem=220GB                 # Request total memory on each node
#SBATCH --time=48:00:00             # Set an 48-hour time limit
#SBATCH --output=/media02/nthuy/LongVidLLaMA/logs/%j.out  # Log output (%j = job ID)
#SBATCH --error=/media02/nthuy/LongVidLLaMA/logs/%j.err   # Log errors (%j = job ID)

# Activate the virtual environment
source /media02/nthuy/Python-3.10.12/thesis_longvu/bin/activate

# Move to the project directory
cd /media02/nthuy/LongVidLLaMA

# Run the training script
bash scripts/entube_finetune_hcmus.sh
