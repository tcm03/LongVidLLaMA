#!/bin/bash
#SBATCH --job-name=imptask          # Job name
#SBATCH --partition=batch           # Partition to use
#SBATCH --nodelist=gpu02,gpu03      # Specify gpu02 and gpu03
#SBATCH --gres=gpu:4                # Request 4 GPUs per node
#SBATCH --ntasks-per-node=1         # One task per node (torchrun manages GPUs)
#SBATCH --cpus-per-task=48          # Request all CPUs on each node (48 CPUs per node)
#SBATCH --mem=244G                  # Request total memory on each node
#SBATCH --time=08:00:00             # Set an 8-hour time limit
#SBATCH --output=/media02/nthuy/LongVidLLaMA/logs/%j.out  # Log output (%j = job ID)
#SBATCH --error=/media02/nthuy/LongVidLLaMA/logs/%j.err   # Log errors (%j = job ID)

# Activate the virtual environment
source /media02/nthuy/Python-3.10.12/thesis_longvu/bin/activate

# Move to the project directory
cd /media02/nthuy/LongVidLLaMA

# Get the master node's address
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=12345

# Run the training script
srun torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT scripts/entube_finetune_hcmus.sh

