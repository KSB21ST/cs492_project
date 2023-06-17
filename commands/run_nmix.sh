# export OMP_NUM_THREADS=1
# git rev-parse --short HEAD
# parallel --eta --ungroup --jobs 120 python main.py --config_file ./configs/RPG.json --config_idx {1} ::: $(seq 1 360)
# parallel --eta --ungroup python main.py --config_file ./configs/Maxmin_minatar_run.json --wandb_mode disabled --config_idx {1} ::: $(seq 1 20)
# python main.py --config_file ./configs/NMIX_catcher_run.json --wandb_mode disabled --config_idx 1
parallel --eta --ungroup --jobs 120 -a commands.txt