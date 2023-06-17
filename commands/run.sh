# Experiment 1 & 2: Run all DQN, DDQN, AveragedDQN, Maxmin avg_mean experiments
parallel --eta --ungroup --jobs 120 -a run_avg.txt

# Experiment 1 & 2: Run all NMIX avg_mean experiments
parallel --eta --ungroup --jobs 120 -a mean_nmix.txt

# Experiment 3: Run all MDP environment
parallel --eta --ungroup python main.py --config_file ./configs/Maxmin_nchain.json --wandb_mode online --device cuda:0 --config_idx {1} ::: $(seq 1 11)