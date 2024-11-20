from train_config import work_list_pairs
import gymnasium as gym
from tqdm import tqdm

trials = 10
episodes = 100
results = {}
for env_id, seed in work_list_pairs:
    env = gym.make(env_id, render_mode = "rgb_array")
    trial_scores = []
    for trial in tqdm(range(trials), desc=f"Env: {env_id}, Seed: {seed}"):
        episode_scores = []
        for episode in range(episodes):
            observation, info = env.reset(seed=seed)
            episode_reward = 0
            done = False
            while not done:
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_scores.append(episode_reward)
        trial_average = sum(episode_scores) / episodes
        trial_scores.append(trial_average)  
    average_score_seed = sum(trial_scores) / trials
    results[(env_id, seed)] = average_score_seed
    env.close()

unique_envs = set(key[0] for key in results.keys())
average_scores = {}
for env in unique_envs:
    env_results = {key: value for key, value in results.items() if key[0] == env}
    average_scores[env] = round(sum(env_results.values()) / len(env_results), 2)

print(average_scores, file=open("mujoco_rand_scores", "a"))