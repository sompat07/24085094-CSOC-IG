import numpy as np 
import gymnasium as gym
import time
import sys

class FrozenLakeEnv():
    def __init__(self, desc=None, is_slippery=0):
        self.desc = desc if desc is not None else self._default_map()
        self.is_slippery = is_slippery
        self.nrow, self.ncol = self.desc.shape
        self.action_space = gym.spaces.Discrete(4) # Up, Down, Left, Right
        self.observation_space = gym.spaces.Discrete(self.nrow * self.ncol)
        self.P = self._generate_transition_probabilities()
        
        
    def _default_map(self):
        return np.array([
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ])
    
    def reset(self,start_state=0):
        self.state = start_state
        r,c = self.state_to_rc(start_state)
        self.step_count = 0
        if self.desc[r,c] == 'H':
            raise ValueError("Starting state cannot be a hole")
        elif self.desc[r,c] == 'G':
            raise ValueError("Starting state cannot be the goal")
        else:
            return self.state

    def get_next_state(self, state, action):
        if action==0: #left
            new_state = state - 1 if state % self.ncol > 0 else state
        elif action==1: #down
            new_state = state + self.ncol if state + self.ncol < self.nrow * self.ncol else state
        elif action==2: #right
            new_state = state + 1 if (state + 1) % self.ncol != 0 else state
        elif action==3: #up
            new_state = state - self.ncol if state - self.ncol >= 0 else state
        else:
            raise ValueError("Invalid action")
        
        return new_state
    
    def state_to_rc(self,state):
        r = state // self.ncol
        c = state % self.ncol
        return r,c
    
    def rc_to_state(self,r,c):
        state = r*self.ncol + c
        return state

    def step(self, action):
        if self.is_slippery > 0:
            action = np.random.choice([action, (action + 1) % 4, (action - 1) % 4], p=[1 - self.is_slippery,self.is_slippery/2,self.is_slippery/2])

        new_state = self.get_next_state(self.state, action)
        
        r,c = self.state_to_rc(new_state)
        # Check if the new state is a hole or goal
        reward = 1 if self.desc[r,c]=='G' else 0
        terminated = (self.desc[r,c]=='H' or self.desc[r,c]=='G')

        self.state = new_state
        self.step_count += 1
        truncated = False
        if self.step_count >= 100:
            self.step_count = 0
            truncated = True
        
        return new_state, reward, terminated, truncated, {}
    
    def _generate_transition_probabilities(self):
        P = {s: {a: [] for a in range(self.action_space.n)} for s in range(self.observation_space.n)}

        for s in range(self.observation_space.n):
            for a in range(self.action_space.n):
                r,c = self.state_to_rc(s)
                if self.desc[r,c] == 'H':
                    P[s][a].append((1.0, s, 0, True))
                elif self.desc[r,c] == 'G':
                    P[s][a].append((1.0, s, 0, True))
                elif self.is_slippery == 0:
                    new_state = self.get_next_state(s, a)
                    r_, c_ = self.state_to_rc(new_state)
                    tile = self.desc[r_,c_]
                    if tile == 'H':
                        reward = 0
                        terminated = True
                    elif tile == 'G':
                        reward = 1
                        terminated = True
                    else:
                        reward = 0
                        terminated = False

                    P[s][a].append((1.0, new_state, reward, terminated))
                else:
                    for action in [a, (a - 1) % 4, (a + 1) % 4]:
                        new_state = self.get_next_state(s, action)  
                        r_, c_ = self.state_to_rc(new_state)
                        prob = 1 - self.is_slippery if action == a else self.is_slippery / 2
                        tile = self.desc[r_,c_]
                        if tile == 'H':
                            reward = 0
                            terminated = True
                        elif tile == 'G':
                            reward = 1
                            terminated = True
                        else:
                            reward = 0
                            terminated = False

                        P[s][a].append((prob,new_state,reward,terminated))
        return P

    def train(self,theta=1e-10, gamma=0.99):
        states = self.observation_space.n
        actions = self.action_space.n
        
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
                                            for prob, next_state, reward, _ in self.P[s][a])
                V[s] = np.max(action_values)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        for s in range(states):
            action_values = np.zeros(actions)
            for a in range(actions):
                action_values[a] = sum(prob * (reward + gamma * V[next_state]) 
                                        for prob, next_state, reward, _ in self.P[s][a])
            policy[s] = np.argmax(action_values)
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        return policy,time_taken
    
    def run(self, policy, episodes=100, start=0):
        time_per_ep = np.zeros(episodes)
        for i in range(episodes):
            start_time = time.perf_counter()

            self.state = self.reset(start)
            while True:
                action = policy[self.state]
                new_state,reward,terminated,truncated,_ = self.step(action)
                
                self.state = new_state
                if terminated or truncated:
                    break
            end_time = time.perf_counter()

            time_per_ep[i] = end_time - start_time
        return time_per_ep

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    env = FrozenLakeEnv(is_slippery=0)
    env2 = gym.make("FrozenLake-v1", desc=env.desc, is_slippery=False, render_mode=None)

    env.P = env2.P.copy()  # Copy transition probabilities from the gym environment    
    policy, ct = env.train(gamma=0.99, theta=1e-10)
    tpe = env.run(policy,episodes=10000,start=0)

    with open("output2.txt","w") as file:
        sys.stdout = file

        print("=====================")
        print(policy.reshape(env.desc.shape))
        print("=====================")
        print(f"Convergence Time = {ct*1000:.4f} ms")
        print("=====================")
        print(f"Mean Time Per Episode = {np.mean(tpe)*1000:.4f} ms")
        print("=====================")
        print(f"Std Dev Time Per Episode = {np.std(tpe)*1000:.4f} ms")
        print("=====================")
        ctr=0
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                ctr += len(env.P[s][a])
        print(f"Total Transitions = {ctr}")
        
        
        file.close()
    sys.stdout = sys.__stdout__