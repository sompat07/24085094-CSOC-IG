import numpy as np
import gymnasium as gym
import time

def line():
  print("==============================================================")

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

def value_iteration(env,theta,gamma):
  states = env.observation_space.n
  actions = env.action_space.n
  
  V = np.zeros(states) #state-value function

  while True:
    delta = 0
    for s in range(states):
      v = V[s]
      q = np.zeros(actions)
      for a in range(actions):
        q[a] = sum(prob*(reward + gamma*V[s_]) for prob,s_,reward,done in env.P[s][a])

      V[s] = np.max(q)
      delta = np.maximum(delta,np.abs(v-V[s]))
    if delta < theta:
      break
  return V
  
def policy_extraction(env,V,gamma):
  states = env.observation_space.n
  actions = env.action_space.n
  
  policy = np.zeros(states)
  
  for s in range(states):
    q = np.zeros(actions)
    for a in range(actions):
      q[a] = sum(prob*(reward + gamma*V[s_]) for prob,s_,reward,done in env.P[s][a])
    
    policy[s] = np.argmax(q)
  
  return policy

def print_policy(env,policy):
    states = env.observation_space.n
    chart = []
    for i in range(env.nrow):
        row = []
        for j in range(env.ncol):
            state = i * env.ncol + j
            action = policy[state]
            if action == 0:  # left
                row.append("←")
            elif action == 1:  # down
                row.append("↓")
            elif action == 2:  # right
                row.append("→")
            elif action == 3:  # up
                row.append("↑")
            else:
                row.append("?")
        chart.append(row)
    print("Optimal Policy:")
    for row in chart:
        print(" ".join(row))

def run(env, optimal_policy):
   state = env.reset()[0]
   while True:
        action = int(optimal_policy[state])
        state, reward, terminated, truncated, info = env.step(action)
       
        if terminated or truncated:
            break
   

if __name__ == "__main__":

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()
    line()
    #Using Policy Iteration Method

    # start1 = time.perf_counter()
    # optimal_value_function, optimal_policy = policy_iteration(env, theta=1e-10, gamma=0.9)
    # end1 = time.perf_counter()

    # convergence_time1 = end1 - start1
    # print(f"Convergence time by policy iteration method: {(convergence_time1 * 1000):.4f} milliseconds")
    # line() 

    # run(env, optimal_policy)

    #Using Value Iteration Method
    ctpi = np.zeros(100)
    elpi = np.zeros(100)
    for i in range(100):
        start2 = time.perf_counter()
        optimal_value_function = value_iteration(env,theta=1e-10,gamma=0.9)
        optimal_policy = policy_extraction(env,optimal_value_function,gamma=0.9)
        end2 = time.perf_counter()

        ctpi[i] = end2 - start2

        start3 = time.perf_counter()
        run(env,optimal_policy)
        end3 = time.perf_counter()

        elpi[i] = end3 - start3
    
    print(ctpi.mean()*1000,'\n',elpi.mean()*1000)


        
    


    