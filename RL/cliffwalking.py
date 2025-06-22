import numpy as np
import gymnasium as gym
import time

def show(q):
    env_render = gym.make('CliffWalking-v0', render_mode='human')
    s = env_render.reset()[0]
    done = False
    while not done:
        a = np.argmax(q[s])
        s_, r, terminated, truncated, info = env_render.step(a)
        done = terminated or truncated
        s = s_
    env_render.close()

class CliffWalking:
    def __init__(self,max_episode_steps=100):
        self.q = np.zeros((48,4))
        self.s = 36
        self.xy = (0,0)
        self.t = 0
        self.mes = max_episode_steps
        self.rg = [] #reached goal or not
        self.rewards = []

        # self.env = gym.make('CliffWalking-v0',render_mode = 'human')

    def reset(self):
        self.s = 36
        self.xy = (0,0)

    def reward_function(self):
        if self.s >= 36:
            if self.s == 47:
                return 100 # prize for finishing
            elif self.s == 36:
                return -2 # slightly more penalty for going back to start
            else:
                return -50  # huge penalty for cliff falling
        return -1 # penalty over timestep

    def step(self,action):
        x,y = self.xy
        if action == 0:
            y = np.minimum(3,y+1) #up
        elif action == 1:
            x = np.minimum(11,x+1) #right
        elif action == 2:
            y = np.maximum(0,y-1) #down
        elif action == 3:
            x = np.maximum(0,x-1) #left
        self.xy = (x,y)
        self.t += 1
        s_ = 36 + x - y*12
        self.s = s_
        r = self.reward_function()
        d = True if self.s > 36 or self.t == self.mes else False
        return s_, r, d

    def run(self,algorithm='q_learning',lr = 0.1,gamma=0.9,episodes=100,epsilon=1,epsilon_decay=0.999,epsilon_final=0.01):
        if algorithm == 'q_learning':
            start = time.perf_counter()
            e = epsilon
            for i in range(episodes):
                self.reset()
                # self.env.reset()
                done = False
                reward = 0
                while not done:
                    s = self.s
                    a = (np.argmax(self.q[s]) if np.random.random() < e else np.random.randint(4))
                    s_, r, done = self.step(a)
                    # _,_,_,_,_ = self.env.step(a)
                    reward += r
                    self.q[s,a] += lr * (r + gamma * np.max(self.q[s_]) - self.q[s,a])
                if self.s == 47 and reward == 88:
                    self.rg.append(i)

                self.rewards.append(reward)
                e = np.maximum(epsilon_final,e*epsilon_decay)

            ct = time.perf_counter() - start

        elif algorithm == 'sarsa':
            start = time.perf_counter()
            e = epsilon
            for i in range(episodes):
                self.reset()
                done = False
                a = (np.argmax(self.q[self.s]) if np.random.random() < e else np.random.randint(4))
                reward = 0
                while not done:
                    s = self.s
                    s_, r, done = self.step(a)
                    a_ = (np.argmax(self.q[self.s]) if np.random.random() < e else np.random.randint(4))
                    self.q[s, a] += lr * (r + gamma * self.q[s_,a_] - self.q[s, a])
                    a = a_
                    reward += r
                if self.s == 47 and reward == 88:
                    self.rg.append(i)
                self.rewards.append(reward)
                e = np.maximum(epsilon_final, e * epsilon_decay)
            ct = time.perf_counter() - start
        else:
            raise NotImplementedError
        policy = np.zeros(48)
        for s in range(48):
            policy[s] = np.argmax(self.q[s])
        # print(policy.reshape(4,12))
        # print('-------------------------------')
        # print(f'Convergence Time: {1000*ct:.2f} ms')
        # print('-------------------------------')
        # print(max(self.rewards))

if __name__ == '__main__':
    ep = 1000

    print('Sarsa:')
    sarsa = CliffWalking(max_episode_steps=100)
    sarsa.run(algorithm='sarsa',episodes=ep)
    q1 = sarsa.q
    print(sum(sarsa.rewards))
    show(q1)

    print('Q_learning:')
    q_learning = CliffWalking(max_episode_steps=100)
    q_learning.run(algorithm='q_learning',episodes=ep)
    q2 = q_learning.q
    print(sum(q_learning.rewards))
    show(q2)

























