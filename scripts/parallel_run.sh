cd Explorer/
# Run jobs
parallel --delay 1 --results parallel_output/ python main.py --config_file ./configs/Catcher.json --config_idx {1} ::: $(seq 1 1024)