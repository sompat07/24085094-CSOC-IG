import numpy as np
import gymnasium as gym
import time
import sys

def initialize(env):
    states = env.observation_space.n
    V = np.zeros(states)  # state-value function
    policy = [env.action_space.sample() for s in range(states)]  # initial policy
    return V, policy

def optimized_policy_iteration(env, theta, gamma):
    states = env.observation_space.n
    actions = env.action_space.n
    V = np.zeros(states)
    policy = np.zeros(states, dtype=int)
    is_policy_stable = False

    while not is_policy_stable:
        # Policy Evaluation (in-place, vectorized)
        while True:
            delta = 0
            for s in range(states):
                v = V[s]
                a = policy[s]
                V[s] = sum(prob * (reward + gamma * V[s_]) for prob, s_, reward, done in env.P[s][a])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement (vectorized)
        is_policy_stable = True
        for s in range(states):
            old_action = policy[s]
            q_sa = np.array([
                sum(prob * (reward + gamma * V[s_]) for prob, s_, reward, done in env.P[s][a])
                for a in range(actions)
            ])
            best_action = np.argmax(q_sa)
            policy[s] = best_action
            if old_action != best_action:
                is_policy_stable = False

        return V, policy
def policy_evaluation(env, policy, theta, gamma):
    states = env.observation_space.n
    V = np.zeros(states)  # state-value function

    while True:
        delta = 0
        for s in range(states):
            v = V[s]
            V[s] = sum(prob * (reward + gamma * V[s_]) for prob, s_, reward, done in env.P[s][policy[s]])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(env,V,policy, gamma):
    states = env.observation_space.n
    actions = env.action_space.n
    
    policy_stable = True
    for s in range(states):
        old_action = policy[s]
        q = np.zeros(actions)
        for a in range(actions):
            q[a] = sum(prob * (reward + gamma * V[s_]) for prob, s_, reward, done in env.P[s][a])
        
        policy[s] = np.argmax(q)
        if old_action != policy[s]:
            policy_stable = False

    return policy, policy_stable

def policy_iteration(env, theta, gamma):
    V, policy = initialize(env)

    while True:
        V = policy_evaluation(env, policy, theta, gamma)
        policy, policy_stable = policy_improvement(env, V, policy, gamma)
        if policy_stable:
            break

    return V, policy

def train(env,theta=1e-10, gamma=0.99):
        states = env.observation_space.n
        actions = env.action_space.n
        
        V = np.zeros(states)
        policy = np.zeros(states, dtype=int)
        start_time = time.perf_counter()
        while True:
            delta = 0
            for s in range(states):
                v = V[s]
                action_values = np.zeros(actions)
                for a in range(actions):
                    action_values[a] = sum(prob * (reward + gamma * V[next_state]) 
                                            for prob, next_state, reward, _ in env.P[s][a])
                V[s] = np.max(action_values)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        for s in range(states):
            action_values = np.zeros(actions)
            for a in range(actions):
                action_values[a] = sum(prob * (reward + gamma * V[next_state]) 
                                        for prob, next_state, reward, _ in env.P[s][a])
            policy[s] = np.argmax(action_values)
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        return policy,time_taken

def run(env,optimal_policy,episodes=100):
    state = env.reset()[0]
    tpe = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        start_time = time.perf_counter()
        while True:
            action = int(optimal_policy[state])
            state, reward, terminated, truncated, info = env.step(action)
        
            if terminated or truncated:
                break
        end_time = time.perf_counter()
        tpe[i] = end_time - start_time
        
    return tpe

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode=None)
    policy, ct = train(env,gamma=0.99, theta=1e-10)
    tpe = run(env,policy,episodes=10000)

    with open("output.txt","w") as file:
        sys.stdout = file
        
        print("=====================")
        print(policy.reshape(env.desc.shape))
        print("=====================")
        print(f"Convergence Time = {ct*1000:.4f} ms")
        print("=====================")
        print(f"Mean Time Per Episode = {np.mean(tpe)*1000:.4f} ms")
        print("=====================")
        

        file.close()
    sys.stdout = sys.__stdout__
