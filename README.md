# Reinforcement Learning Agents Parallel Training and Subset Selection

This repository contains scripts for training RL agents in parallel across two GPUs, calculating random agent performance, and identifying the best subsets of environments for efficient multi-benchmark testing. The research and experiments performed here are part of my thesis on reinforcement learning benchmark generalization. https://utheses.univie.ac.at/detail/73129/



## File Descriptions

### `gpu_parallel.py`
- This script is used for training reinforcement learning agents in parallel using two GPUs. Each GPU runs 12 processes in parallel.
- It relies on two additional files:
  - `train_config.py`: Specifies which environments to train, which algorithms to use, and which seeds to run the experiments over.
  - `train_functions.py`: Provides all the functions required for training the agents using [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3). Additionally, it logs training progress using TensorBoard and weights and biases for reward tracking.
- The trained agents' scores and WandB links are stored in the `logs` folder.

### `rand_agent.py`
- A script to obtain the random agent’s score for a Gymnasium RL environment.

### `yml` folder
- Contains the hyperparameter configurations used during the training processes.

### `dataset.csv`
- Created using `scores_means.ipynb`, this evaluates mean scores of algorithm performance, averaged across four different seeds from the log folders. 

### `subsetsearch2.ipynb`
- This notebook preprocesses and normalizes the data from `dataset.csv`, then runs regression models to find the best subsets of environments.
- It also performs evaluation experiments and includes the case study experiment as discussed in the thesis.

## How to Use

1. To train RL agents in parallel, configure the `train_config.py` file to set environments, algorithms, and seeds. Run `gpu_parallel.py` to start training.
2. Use `rand_agent.py` to calculate random agent scores for a specific environment.
3. After training, use `scores_means.ipynb` to get the mean scores for environments over different seeds for algorithms. 
4. Finally, use `subsetsearch2.ipynb` to preprocess the dataset, run regression models, and analyze the results for the case study.
