#!/bin/bash
#SBATCH --job-name=atari_ram_3
#SBATCH --account=rrg-whitem
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=output/%x/%j.txt
#SBATCH --mail-user=qlan3@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --mail-type=TIME_LIMIT

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
module load singularity/2.6
singularity exec -B /project ../explorer-env.img python main.py --config_file ./configs/atari_ram_3.json --config_idx $SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------