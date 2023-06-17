# NMIX Q Learning

Code based from [Explorer](https://github.com/qlan3/Explorer).


## Implemented algorithms

- Vanilla Deep Q-learning (VanillaDQN): No target network.
- [Deep Q-Learning (DQN)](https://users.cs.duke.edu/~pdinesh/sources/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double Deep Q-learning (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Maxmin Deep Q-learning (MaxminDQN)](https://arxiv.org/pdf/2002.06487.pdf)
- [Averaged Deep Q-learning (AveragedDQN)](https://arxiv.org/pdf/1611.01929.pdf)


## The dependency tree of agent classes

    Base Agent
      └── Vanilla DQN
            ├── DQN
            |    └──DDQN
            ├── Maxmin DQN
            ├── Averaged DQN
            └── NMIX DQN


## Requirements

- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- [Gym && Gym Games](https://github.com/qlan3/gym-games): You may only install part of Gym (`classic_control, box2d`) by command `pip install 'gym[classic_control, box2d]'`.
- Optional: 
  - [Gym Atari](https://www.gymlibrary.ml/environments/atari/): `pip install gym[atari,accept-rom-license]`
  - [Gym Mujoco](https://www.gymlibrary.ml/environments/mujoco/):
    - Download MuJoCo version 1.50 from [MuJoCo website](https://www.roboti.us/download.html).
    - Unzip the downloaded `mjpro150` directory into `~/.mujoco/mjpro150`, and place the activation key (the `mjkey.txt` file downloaded from [here](https://www.roboti.us/license.html)) at `~/.mujoco/mjkey.txt`.
    - Install [mujoco-py](https://github.com/openai/mujoco-py): `pip install 'mujoco-py<1.50.2,>=1.50.1'`
    - Install gym[mujoco]: `pip install gym[mujoco]`
  - [PyBullet](https://pybullet.org/): `pip install pybullet`
  - [DeepMind Control Suite](https://github.com/denisyarats/dmc2gym): `pip install git+git://github.com/denisyarats/dmc2gym.git`
- Others: Please check `requirements.txt`.


## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict.

For example, run the experiment with configuration file `Maxmin_catcher_run.json` and configuration index `1`:

```python main.py --config_file ./configs/Maxmin_catcher_run.json --config_idx 1```

The models are tested for one episode after every `test_per_episodes` training episodes which can be set in the configuration file.


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `Maxmin_catcher_run.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in Maxmin_catcher_run.json: 55`

Then we run through all configuration indexes from `1` to `55`. The simplest way is using a bash script:

``` bash
for index in {1..55}
do
  python main.py --config_file ./configs/Maxmin_catcher_run.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/Maxmin_catcher_run.json --config_idx {1} ::: $(seq 1 55)
```
For more information, please check the `run.sh` file and `commands/` folder.
# Environment installation

Environment based from [gym-games](https://github.com/qlan3/gym-games).

``` bash
cd gym-games
python setup.py install
```

# Acknowledgements

- [Explorer](https://github.com/qlan3/Explorer)
- [DeepRL](https://github.com/ShangtongZhang/DeepRL)
- [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
- [Classic Control](https://github.com/muhammadzaheer/classic-control)
- [Spinning Up in Deep RL](https://github.com/openai/spinningup)
- [Randomized Value functions](https://github.com/facebookresearch/RandomizedValueFunctions)
- [Rainbow](https://github.com/Kaixhin/Rainbow)
