import sys
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import pickle
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def line():
    print("==============================================================")

def reward_function(s,s_, reward, terminated,truncated):
    if terminated or truncated:
        if reward:
            reward = 10
        else:
            reward = -10
    else:
        if s_ > s:
            reward = (s_//m + s_%m)/(m**2)
        else:
            reward = -(s//m + s%m)/(m**2)

    return reward

def value_iteration(env,p):
    states = env.observation_space.n
    actions = env.action_space.n
    env = env.unwrapped
    theta = p["theta"]
    gamma = p["discount_factor"]
    V = np.zeros(states)
    policy = np.zeros(states, dtype=int)
    start = time.perf_counter()
    while True:
        delta = 0
        for s in range(states):
            v = V[s]
            q = np.zeros(actions)
            for a in range(actions):
                q[a] = sum(prob * (reward + gamma * V[s_]) for prob, s_, reward, done in env.P[s][a])
            V[s] = max(q)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    Q = np.zeros((states,actions))
    for s in range(states):
        for a in range(actions):
            Q[s,a] = sum(prob * (reward + gamma * V[s_]) for prob, s_, reward,
            done in env.P[s][a])

    return Q, time.perf_counter() - start

def monte_carlo(env,p):
    states = env.observation_space.n
    actions = env.action_space.n
    # load important parameters
    desc = p['desc']
    episodes = p["episodes"]
    gamma = p["discount_factor"]
    epsilon = p["epsilon"]
    # initialization
    Q = np.random.rand(states, actions)
    policy = np.full((states, actions), 0.25)
    returns = {s: {a: [] for a in range(actions)} for s in range(states)}
    # iteration
    rpe = np.zeros(episodes)
    start = time.perf_counter()
    for i in range(episodes):
        s = env.reset()[0]
        episode = []
        rewards = []
        done = False
        # generate an episode

        while not done:
            a = np.random.choice([0,1,2,3],p=policy[s])
            s_, reward, terminated, truncated, _ = env.step(a)
            reward = reward_function(s,s_, reward, terminated,truncated)
            done = terminated or truncated
            episode.append((s, a))
            rewards.append(reward)
            s = s_
        rpe[i] = rewards[-1]
        # traverse the episode in reverse to calculate returns
        G = 0
        saG = []
        for t in range(len(episode)-1,-1,-1):
            state, action = episode[t]
            reward = rewards[t]
            G = gamma * G + reward
            saG.append((state,action,G))

        seen = set()
        for t in range(len(saG) - 1, -1, -1):
            state,action,G = saG[t]
            if (state, action) not in seen: #first-visit
                seen.add((state, action))
                returns[state][action].append(G)
                Q[state,action] = np.mean(returns[state][action])
                A = np.argmax(Q[state,:])
                for a in range(actions):
                    policy[state,a] = 1 - epsilon + epsilon / actions if a == A else epsilon / actions
                policy[state] /= sum(policy[state])

        if i % (episodes // 100) == 0:
            print(f"{i // (episodes // 100)}% done. Final state: {s} Reward: "
                  f"{rpe[i]}")

    return Q,time.perf_counter() - start

def SARSA(env,p):
    states = env.observation_space.n
    actions = env.action_space.n
    #load important parameters
    episodes = p["episodes"]
    gamma = p["discount_factor"]
    epsilon = p["epsilon"]
    alpha = p['learning_rate']
    edr = p['epsilon_decay_rate']
    #initialization
    Q = np.random.rand(states, actions)
    policy = np.zeros(states)

    #iteration
    rpe = np.zeros(episodes)
    start = time.perf_counter()
    for i in range(episodes):
        s = env.reset()[0]
        done = False
        a = env.action_space.sample()
        while not done:
            s_, reward, terminated, truncated, _ = env.step(a)
            if np.random.rand() < epsilon:
                a_ = env.action_space.sample()
            else:
                a_ = np.argmax(Q[s])
            reward = reward_function(s,s_,reward,terminated,
                                     truncated)
            Q[s,a] += alpha * (reward + gamma * Q[s_,a_] - Q[s,a])
            done = terminated or truncated
            s = s_
            a = a_
        rpe[i] = reward
        epsilon -= edr
        if i % (episodes // 100) == 0:
            print(f"{i // (episodes // 100)}% done. Final state: {s} Reward: "
                  f"{rpe[i]}")

    return Q, time.perf_counter() - start

def q_learning(env,p):
    states = env.observation_space.n
    actions = env.action_space.n
    # load important parameters
    episodes = p["episodes"]
    gamma = p["discount_factor"]
    epsilon = p["epsilon"]
    alpha = p['learning_rate']
    edr = p['epsilon_decay_rate']
    # initialization
    Q = np.random.rand(states, actions)
    policy = np.zeros(states)

    #iteration
    rpe = np.zeros(episodes)
    start = time.perf_counter()
    for i in range(episodes):
        s = env.reset()[0]
        done = False

        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s])
            s_, reward, terminated, truncated, _ = env.step(a)
            reward = reward_function(s,s_,reward,terminated,
                                     truncated)
            Q[s,a] += alpha * (reward + gamma * np.max(Q[s_]) - Q[s,a])
            done = terminated or truncated
            s = s_
        rpe[i] = reward
        epsilon -= edr
        if i % (episodes // 100) == 0:
            print(f"{i // (episodes // 100)}% done. Final state: {s} Reward: "
                  f"{rpe[i]}")

    return Q, time.perf_counter() - start

def train(env,algorithm,p):
    if algorithm == "policy_iteration":
        Q, ct = policy_iteration(env, p)
    elif algorithm == "value_iteration":
        Q, ct = value_iteration(env, p)
    elif algorithm == "monte_carlo":
        Q, ct = monte_carlo(env, p)
    elif algorithm == "SARSA":
        Q, ct = SARSA(env, p)
    elif algorithm == "q_learning":
        Q, ct = q_learning(env, p)
    else:
        raise ValueError("Unknown algorithm: {}".format(algorithm))

    return Q, ct

def show(lake,q,ep=1):
    env_render = gym.make('FrozenLake-v1',desc=lake,is_slippery=False,
                          render_mode='human',max_episode_steps=500)
    tpe = []
    for i in range(ep):
        start = time.perf_counter()
        s = env_render.reset()[0]
        done = False
        while not done:
            a = np.argmax(q[s])
            s_, r, terminated, truncated, info = env_render.step(a)
            done = terminated or truncated
            s = s_
        tpe.append(time.perf_counter() - start)
        env_render.close()

    return np.mean(tpe)

if __name__ == "__main__":
    m = 20
    lake = generate_random_map(size=m, p=0.9, seed=0)

    render = False
    env = gym.make('FrozenLake-v1', desc=lake, is_slippery=False,
                   max_episode_steps=500, render_mode='human' if render else
        None)
    ep = 100000
    p = {
        'desc': lake,
        'length': m,
        "discount_factor": 0.99,
        'theta':1e-10,
        "epsilon": 1,
        "epsilon_decay_rate": 0.8/ep,
        'episodes': ep,
        'learning_rate': 0.01
    }
    for algo in [
        # "value_iteration",
        "monte_carlo",
        "SARSA",
        "q_learning"
    ]:
        print(f'Algorithm: {algo}')
        print('----------------------------')
        q, ct = train(env,algo,p)
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            policy[s] = np.argmax(q[s])

        print(policy.reshape(m, m))
        print('----------------------------')
        print(f'convergence time: {ct*1000:.4f} ms')
        print('-----------------------------')
        mel = show(lake, q)
        print(f'mean episode length: {mel*1000:.4f} ms')
        print('-----------------------------')
        pickle.dump(policy, open(algo + '_policy.pkl', 'wb'))