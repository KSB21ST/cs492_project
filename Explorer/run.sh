# for index in $(seq 1 1);
# do
#   python main.py --config_file ./configs/NMix_catcher.json --config_idx $index
# done
parallel --eta --ungroup python main.py --config_file ./configs/nchain_custom.json --wandb_mode online --device cuda:0 --config_idx {1} ::: $(seq 1 4)