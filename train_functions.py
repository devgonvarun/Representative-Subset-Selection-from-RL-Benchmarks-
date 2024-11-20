import os
import csv
import time
import wandb
import yaml
import shutil
import torch.nn as nn
from torch.optim import RMSprop
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from train_config import benchmark_mapping
from datetime import datetime

algo_mapping = {"PPO": PPO, "A2C": A2C, "TRPO": TRPO, "RecurrentPPO": RecurrentPPO}

def get_benchmark_name(env_id):
    return benchmark_mapping.get(env_id, "Unknown")

def get_hyperparams(env_id, algo):
    with open("yml/" + algo + ".yml", 'r') as f:
        hyperparams = yaml.safe_load(f)
        if (get_benchmark_name(env_id)=="atari"):
            return hyperparams.get("atari", {})
        else:
            return hyperparams.get(env_id, {})


def get_policy(env_id, algo):
    if (get_benchmark_name(env_id)=="atari"):
        if algo == "RecurrentPPO":
            return "CnnLstmPolicy"
        else:
            return "CnnPolicy"
    elif (get_benchmark_name(env_id)=="dm_control"):
        if algo == "RecurrentPPO":
            return "MultiInputLstmPolicy"
        else:
            return "MultiInputPolicy"
    else:
        if algo == "RecurrentPPO":
            return "MlpLstmPolicy"
        else:
            return "MlpPolicy"


def get_policy_args(env_id, algo):
    if algo=="PPO" and env_id in ["HalfCheetah-v4", "CheetahRun", "Ant-v4", "Hopper-v4"]:
        policy_args = dict(log_std_init = -2, ortho_init = False, activation_fn = nn.ReLU ,net_arch = dict(pi=[256, 256], vf=[256, 256]))
    elif algo=="A2C" and get_benchmark_name(env_id)=="atari":
        policy_args = dict(optimizer_class=RMSprop, optimizer_kwargs=dict(eps=1e-5))
    elif algo=="RecurrentPPO" and get_benchmark_name(env_id)=="atari":
        policy_args = dict(enable_critic_lstm=False, lstm_hidden_size=128)
    else:
        policy_args = None
    return policy_args

def initialize_wandb(env_id):
    config = {
        #"total_timesteps": int(1e10),
        "total_timesteps": 1_000_000,
        "env_id": env_id,
    }
    return wandb.init(
        project="atari",
        config=config,
        sync_tensorboard=True,
        settings=wandb.Settings(silent="true", _disable_stats=True)
    )

def log_active_task(gpu_id, env_id, seed, algo, time, wandb_url):
    """Using Google Gemini"""
    results_file = "ACTIVE_TRAININGS.csv"
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["GPU", "Environment", "Seed", "Algo", "Start Time", "Wandb Link"])
        writer.writerow([gpu_id, env_id, seed, algo, time, wandb_url])

def del_inactive_task(wandb_url_to_delete):
    """Using Google Gemini"""
    results_file = "ACTIVE_TRAININGS.csv"
    with open(results_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        rows = list(reader)  # Read all rows into a list
    with open(results_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for row in rows:
            if row[5] != wandb_url_to_delete:
                writer.writerow(row)

def log_results(gpu_id, env_id, seed, algo, training_time, trained_until, mean_reward, wandb_url):
    """Logs training results to a CSV file. using Google Gemini Source: github.com/CppMaster/SC2-AI"""
    results_file = "training_results.csv"
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["GPU", "Benchmark", "Environment", "Seed", "Algo", "Training Time", "Trained Until", "Best Mean Reward", "Wandb Link"])
        writer.writerow([gpu_id, get_benchmark_name(env_id), env_id, seed, algo, training_time, trained_until, mean_reward, wandb_url])

def del_active_training_csv():
    os.remove("ACTIVE_TRAININGS.csv")

def get_timestamp():
    return datetime.now().strftime("%H:%M_%dth%b")

def rename_results_csv(timestamp):
    new_name = f'{"RES"}_{timestamp}.csv'
    os.rename("training_results.csv", new_name)
    return new_name

def move_to_logs(gpu_log, saved_results):
    os.makedirs("logs", exist_ok=True)
    shutil.move(gpu_log, os.path.join("logs", gpu_log))
    shutil.move(saved_results, os.path.join("logs", saved_results))

def train_until_convergence(gpu_id, env_id, seed, algo):
    run = initialize_wandb(env_id)
    wandb_url = run.get_url()
    log_active_task(gpu_id=gpu_id, env_id=env_id, seed=seed, algo=algo, time=get_timestamp(), wandb_url=wandb_url)
    n_env = 8
    if (get_benchmark_name(env_id)=="atari"):
        env = make_atari_env(env_id, n_envs=n_env, seed=seed)
        env = VecFrameStack(env, n_stack=4)
        eval_env = make_atari_env(env_id, n_envs=n_env, seed=seed)
        eval_env = VecFrameStack(eval_env, n_stack=4)
    else:
        env = make_vec_env(env_id, n_envs=n_env, seed=seed)
        eval_env = make_vec_env(env_id, n_envs=n_env, seed=seed)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=25, min_evals=100, verbose=1) 
    eval_callback = EvalCallback(eval_env, eval_freq=10_000, callback_after_eval=stop_train_callback, verbose=1) 
    policy = get_policy(env_id, algo)
    policy_kwargs = get_policy_args(env_id, algo)
    hyperparams = get_hyperparams(env_id, algo)
    model = algo_mapping[algo](policy=policy , env=env, verbose=0, device="cuda", tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs, **hyperparams)
    start = time.time()
    model.learn(run.config.total_timesteps*n_env, callback=CallbackList([eval_callback, WandbCallback(model_save_path=None, verbose=0)])) #f"models/{run.id}"
    training_time = round(time.time() - start, 2)
    mean_reward = eval_callback.best_mean_reward
    run.finish()
    env.close()
    del_inactive_task(wandb_url)
    log_results(gpu_id, env_id, seed, algo, training_time, eval_callback.num_timesteps/n_env, mean_reward, wandb_url)


def train_until_timesteps(gpu_id, env_id, seed, algo):
    run = initialize_wandb(env_id)
    wandb_url = run.get_url()
    log_active_task(gpu_id=gpu_id, env_id=env_id, seed=seed, algo=algo, time=get_timestamp(), wandb_url=wandb_url)
    if (get_benchmark_name(env_id)=="atari"):
        if algo == "A2C":
            n_env = 16
        else:
            n_env = 8
        env = make_atari_env(env_id, n_envs=n_env, seed=seed)
        env = VecFrameStack(env, n_stack=4)
    else:
        if algo == "TRPO":
            n_env = 2
        elif algo == "RecurrentPPO":
            n_env = 1
        else:
            n_env = 8
        env = make_vec_env(env_id, n_envs=n_env, seed=seed, env_kwargs={"render_mode": "rgb_array"})
    policy = get_policy(env_id, algo)
    policy_kwargs = get_policy_args(env_id, algo)
    hyperparams = get_hyperparams(env_id, algo)
    model = algo_mapping[algo](policy=policy , env=env, verbose=0, device="cuda", policy_kwargs=policy_kwargs, **hyperparams)
    start = time.time()
    model.learn(run.config.total_timesteps*n_env, callback=WandbCallback(model_save_path=None, verbose=0), progress_bar=True)
    training_time = round(time.time() - start, 2)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    run.finish()
    env.close()
    del_inactive_task(wandb_url)
    log_results(gpu_id, env_id, seed, algo, training_time, run.config.total_timesteps, mean_reward, wandb_url)


def train_on_gpu(queue, gpu_id, queue_lock, log_filename):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with queue_lock:
        while not queue.empty():
            env_id, seed, algo = queue.get()
            queue_lock.release()  
            print(f"GPU {gpu_id}: {env_id} #{seed} {algo} Running      {get_timestamp()}", file=open(log_filename, "a"))
            #train_until_convergence(gpu_id=gpu_id, env_id=env_id, seed=seed, algo=algo)
            train_until_timesteps(gpu_id=gpu_id, env_id=env_id, seed=seed, algo=algo)
            print(f"GPU {gpu_id}: {env_id} #{seed} {algo} Finished     {get_timestamp()}", file=open(log_filename, "a"))
            queue_lock.acquire()
