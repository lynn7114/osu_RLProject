from song_manager import select_song
from beatmap import extract_osz, parse_osu_file
from environment import RhythmEnv
from evaluate import main as evaluate_main

# libraries
import gymnasium as gym
import collections
import random
import numpy as np
import os
import pandas as pd
import typer
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from models.ppo_model import PPO, T_horizon
from models.dqn_models import *
from models.a2c_model import run_A2C

app = typer.Typer(help="Osu! RL training")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_with_seeds(alg_name, render, selected_osu):
    seeds = [0, 23, 147, 575, 2768]
    results = []

    for seed in seeds:
        print(f"\n Running {alg_name} (seed={seed})")
        set_seed(seed)
        score = run_experiment(algorithm_type=alg_name, render=render, selected_osu=selected_osu)
        if score is not None:
            results.append(score)

    if not results:
        print("no results recorded.")
        return

    # 기본 통계 계산
    mean_score = np.mean(results)
    std_score = np.std(results)
    sem = std_score / np.sqrt(len(results))
    ci = 1.96 * sem  # 95% 신뢰구간
    print(f"\n===== {alg_name} summary =====")
    print(f" Seeds: {seeds}")
    print(f" Mean score: {mean_score:.2f}")
    print(f" Std: {std_score:.2f}")
    print(f" 95% CI: [{mean_score - ci:.2f}, {mean_score + ci:.2f}]")

def run_experiment(algorithm_type="PPO", render=False, selected_osu=None, lr=1e-3, gamma=0.99):
    """Run experiment with specified algorithm type"""
    is_dqn = True
    print(f"\n=== Running {algorithm_type} Experiment ===")

    # selected_osu가 None이면 자동으로 선택하게 함 
    if selected_osu is None:
        from song_manager import select_song
        selected_osu = select_song("data")
        if not selected_osu:
            print("No song selected. Exiting experiment.")
            return

    notes = parse_osu_file(selected_osu)
    print(f"Loaded beatmap: {selected_osu} | Notes: {len(notes)}")
    env = RhythmEnv(notes)

    if render:
        notes = parse_osu_file(selected_osu)
        print(f"Loaded beatmap: {selected_osu} | Notes: {len(notes)}")
        env = RhythmEnv(notes)

    if algorithm_type == "Dueling_DQN":
        q = DuelingQnet()
        q_target = DuelingQnet()
        train_fn = train_double_dqn  # Dueling uses Double DQN training
    elif algorithm_type == "PPO":
        model = PPO()
        is_dqn = False
    elif algorithm_type == "A2C":
        run_A2C(env)
        return
    else:
        q = Qnet()
        q_target = Qnet()
        train_fn = train_dqn if algorithm_type == "DQN" else train_double_dqn


    if is_dqn:
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer()
        optimizer = optim.Adam(q.parameters(), lr=lr)

    print_interval = 20
    score = 0.0

    scores = []
    episodes = []

    for n_epi in range(3000):
        s, _ = env.reset()
        done = False

        if is_dqn:
            epsilon = max(0.01, 0.1 * np.exp(-n_epi / 300))
        else:
            episode_reward = 0.0

        while not done:
            if is_dqn:
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, terminated, truncated, info = env.step(a)
                done = (terminated or truncated)
                done_mask = 0.0 if done else 1.0
                memory.put((s,a,r/100.0,s_prime, done_mask))
                s = s_prime

                score += r
                if done:
                    break
            else:
                for t in range(T_horizon):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, terminated, truncated, info = env.step(a)
                    done = terminated or truncated

                    model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                    s = s_prime

                    score += r
                    episode_reward += r
                    if done:
                        break

                model.train_net()
        if is_dqn:
            if memory.size()>2000:
                train_fn(q, q_target, memory, optimizer)

            if n_epi%print_interval==0 and n_epi!=0:
                q_target.load_state_dict(q.state_dict())
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                                n_epi, score/print_interval, memory.size(), epsilon*100))
                score = 0.0
        else:
            scores.append(episode_reward)
            episodes.append(n_epi)

            if n_epi%print_interval==0 and n_epi!=0:
                print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
                score = 0.0

    env.close()

def main():
    selected_osu = select_song("data")  
    if not selected_osu:
        print("A song is not selected.\n")
        return

    """Run experiments for all three algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN", "PPO", "A2C"]
    seeds = [0, 23, 147, 575, 2768]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. PPO")
    print("5. A2C")
    print("6. Run all algorithms")

    choice = input("Enter your choice (1-6): ")

    # Ask about rendering
    render_choice = input("Enable GUI visualization? (y/n): ").lower()
    render = render_choice in ['y', 'yes']

    if choice == "1":
        run_with_seeds("DQN", render, selected_osu)
    elif choice == "2":
        run_with_seeds("Double_DQN", render, selected_osu)
    elif choice == "3":
        run_with_seeds("Dueling_DQN", render, selected_osu)
    elif choice == "4":
        run_with_seeds("PPO", render, selected_osu)
    elif choice == "5":
        run_with_seeds("A2C", render, selected_osu)
    elif choice == "6":
        for alg in algorithms:
            run_with_seeds(alg, render, selected_osu)
    else:
        print("Invalid choice, running DQN by default.")
        run_with_seeds("DQN", render, selected_osu)

@app.command()
def train(
    algo: str = typer.Option("--algo", "PPO", help="Algorithm"),
    lr: float = typer.Option("--learning-rate", 1e-3, help="Learning rate"),
    gamma: float = typer.Option("--gamma", 0.99, help="Discount factor"),
    song: str = typer.Option("--song", None, help="Song number or path"),
    render: bool = typer.Option("--render", False, help="Enable GUI"),
):

    if song is None:
        selected_osu = select_song("data")
    else:
        try:
            # 숫자로 입력된 경우 data 폴더 내 순서에 따라 파일 선택
            idx = int(song) - 1
            osz_files = sorted(glob.glob("data/*.osz"))
            out_dir = extract_osz(osz_files[idx])
            osu_files = sorted(glob.glob(out_dir + "/*.osu"))
            selected_osu = osu_files[0]
        except Exception:
            selected_osu = song  # 직접 경로 입력 경우

    run_with_seeds(algo, render, selected_osu, lr, gamma)

@app.command()
def evaluate_all():
    evaluate_main() 

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        app()   # argument가 있으면 typer CLI 사용
    else:
        main()  # argument 없으면 기존 메뉴모드 사용