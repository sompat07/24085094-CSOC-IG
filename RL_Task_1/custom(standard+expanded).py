import numpy as np 
import gymnasium as gym
import time

class FrozenLakeEnv(gym.Env):
    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        self.desc = desc if desc is not None else self._generate_map(map_name)
        self.is_slippery = is_slippery
        self.nrow, self.ncol = self.desc.shape
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Discrete(self.nrow * self.ncol)
        self.P = self._generate_transition_probabilities()
        
    def _generate_map(self, map_name):
        if map_name == "4x4":
            return np.array([
                ['S', 'F', 'H', 'F'],
                ['H', 'F', 'F', 'H'],
                ['F', 'F', 'F', 'F'],
                ['F', 'H', 'F', 'G']
            ])
        elif map_name == "8x8":
            return np.array([
                ['S', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
                ['F', 'H', 'F', 'H', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'H', 'F', 'H', 'F', 'F'],
                ['F', 'H', 'F', 'F', 'F', 'H', 'F', 'F'],
                ['F', 'F', 'H', 'F', 'F', 'F', 'H', 'F'],
                ['H', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'H', 'F', 'H', 'F'],
                ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'G']])
        else:   
            raise ValueError("Unknown map name")
    
    def reset(self,start_state=0):
        self.state = start_state
        self.rc = self.state_to_rc(start_state)
        self.step_count = 0
        if self.desc[self.state // self.ncol, self.state % self.ncol] == 'H':
            raise ValueError("Starting state cannot be a hole")
        elif self.desc[self.state // self.ncol, self.state % self.ncol] == 'G':
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
        if self.is_slippery:
            action = np.random.choice([action, (action + 1) % 4, (action - 1) % 4], p=[0.34,0.33,0.33])

        new_state = self.get_next_state(self.state, action)
        
        r,c = self.state_to_rc(new_state)
        # Check if the new state is a hole or goal
        if self.desc[r,c] == 'H':
            reward = -1
            terminated = True
        elif self.desc[r,c] == 'G':
            reward = 1
            terminated = True
        else:
            reward = 0
            terminated = False

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
                for action in [a, (a + 1) % 4, (a - 1) % 4]:
                    new_state = self.get_next_state(s, action)  

                    if self.desc[new_state // self.ncol, new_state % self.ncol] == 'H':
                        reward = -1
                        terminated = True
                    elif self.desc[new_state // self.ncol, new_state % self.ncol] == 'G':
                        reward = 1
                        terminated = True
                    else:
                        reward = 0
                        terminated = False

                    P[s][a].append((0.33,new_state,reward,terminated))
        return P

    def train(self, gamma=0.99, theta=1e-10):
        states = self.observation_space.n
        actions = self.action_space.n
        
        V = np.zeros(states) #state-value function
        start_time = time.perf_counter()
        while True:
            delta = 0
            for s in range(states):
                v = V[s]
                q = np.zeros(actions)
                for a in range(actions):
                    q[a] = sum(prob*(reward + gamma*V[s_]) for prob,s_,reward,done in self.P[s][a])

            V[s] = np.max(q)
            delta = np.maximum(delta,np.abs(v-V[s]))
            if delta < theta:
                break

        policy = np.zeros(states)
  
        for s in range(states):
            q = np.zeros(actions)
            for a in range(actions):
                q[a] = sum(prob*(reward + gamma*V[s_]) for prob,s_,reward,done in self.P[s][a])
            
            policy[s] = np.argmax(q)
        end_time = time.perf_counter()
        convergence_time = end_time - start_time
        return policy,convergence_time

    def run(self,policy,episodes=100,start=0,render=False):
        time_per_ep = np.zeros(episodes)

        for i in range(episodes):
            start_time = time.perf_counter()

            self.state = self.reset(start)
            while True:
                action = policy[self.state]
                new_state,reward,terminated,truncated,_ = env.step(action)
                
                self.state = new_state
                if render:
                    self.render()
                if terminated or truncated:
                    break
            
            end_time = time.perf_counter()

            time_per_ep[i] = end_time - start_time
        return time_per_ep

    def render(self, mode='human'):
        r,c = self.state_to_rc(self.state)
        desc_copy = np.copy(self.desc)
        desc_copy[r,c] = 'P'
        with open("output.txt","w") as file:
            print(desc_copy)
        file.close()








    
        
