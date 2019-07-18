#!/bin/bash
#SBATCH --job-name=penn1
#SBATCH --account=def-afyshe-ab
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --output=output/%x/%j.txt
#SBATCH --mail-user=qlan3@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --mail-type=TIME_LIMIT_80

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
module load singularity/2.6
singularity exec -B /project ../gvfn-env.img python main.py --config_file ./configs/PennTreebank_1.json --config_idx $SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------