import numpy as np
import gymnasium as gym
import random


env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n


gamma = 0.99
theta = 1e-6
alpha = 0.1
epsilon = 0.1
episodes = 10000
k = 10  

def value_iteration(env, gamma=0.99, theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            Q_values = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(n_actions)]
            V[s] = max(Q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        Q_values = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(n_actions)]
        policy[s] = np.argmax(Q_values)
    return policy, V


def policy_evaluation(policy, env, gamma=0.99, theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            a = policy[s]
            V[s] = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=0.99, theta=1e-6):
    policy = np.random.choice(n_actions, size=n_states)
    while True:
        V = policy_evaluation(policy, env)
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            Q_values = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(n_actions)]
            policy[s] = np.argmax(Q_values)
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return policy, V


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=10000):
    Q = np.zeros((n_states, n_actions))
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
    policy = np.argmax(Q, axis=1)
    return policy, Q


class EpsilonGreedyAgent:
    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

class UCBAgent:
    def __init__(self, k, c=2):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t += 1
        if 0 in self.N:
            return np.argmin(self.N)
        else:
            return np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

def main():
   
    policy_vi, V_vi = value_iteration(env)
    print("Value Iteration - Optimal Policy:", policy_vi)
    print("Value Iteration - Value Function:", V_vi)

 
    policy_pi, V_pi = policy_iteration(env)
    print("Policy Iteration - Optimal Policy:", policy_pi)
    print("Policy Iteration - Value Function:", V_pi)

   
    policy_ql, Q_ql = q_learning(env)
    print("Q-Learning - Optimal Policy:", policy_ql)
    print("Q-Learning - Q-Table:", Q_ql)

    
    agent_eg = EpsilonGreedyAgent(k)
    rewards_eg = np.random.randn(1000, k)
    for t in range(1000):
        action = agent_eg.select_action()
        reward = rewards_eg[t, action]
        agent_eg.update(action, reward)
    print("Epsilon-Greedy - Estimated Q-values:", agent_eg.Q)

    
    agent_ucb = UCBAgent(k)
    rewards_ucb = np.random.randn(1000, k)
    for t in range(1000):
        action = agent_ucb.select_action()
        reward = rewards_ucb[t, action]
        agent_ucb.update(action, reward)
    print("UCB - Estimated Q-values:", agent_ucb.Q)

if __name__ == "__main__":
    main()
