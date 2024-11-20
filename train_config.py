import csv

algos = ["A2C"] #Add algo names to be trained

#Mujoco     #Add list of environments to train
benchmark_mapping = {
    "dm_control/ball_in_cup-catch": "Mujoco",
    "dm_control/cartpole-balance": "dm_control",
    "dm_control/cartpole-balance_sparse": "dm_control",
    "dm_control/cartpole-swingup": "dm_control",
    "dm_control/cartpole-swingup_sparse": "dm_control",
    "dm_control/cheetah-run": "dm_control",
    "dm_control/finger-spin": "dm_control",
    "dm_control/hopper-hop": "dm_control",
    "dm_control/hopper-stand": "dm_control",
    "dm_control/pendulum-swingup": "dm_control",
    "dm_control/reacher-easy": "dm_control",
    "dm_control/reacher-hard": "dm_control",
}

env_list = list(benchmark_mapping.keys())

seed_list = [111,222,333,444]   #Add env seeds to train on

work_list_pairs = [(env, seed) for env in env_list for seed in seed_list] 

work_list_full = [(env, seed, algo) for env in env_list for seed in seed_list for algo in algos] 

# Read completed experiments from CSV
completed_experiments = set()
try:
    with open("training_results.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            completed_experiments.add((row["Environment"], int(row["Seed"]), row["Algo"]))
except FileNotFoundError:
    print("Warning: 'training_results.csv' not found. Assuming all experiments are new.")

# Exclude completed experiments from work_list
work_list = [exp for exp in work_list_full if exp not in completed_experiments]
