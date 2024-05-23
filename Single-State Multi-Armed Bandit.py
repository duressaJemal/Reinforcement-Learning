import numpy as np
import random


K = 10

N = 1000

reward_distributions = [(np.random.rand(), np.random.rand() * 0.1) for _ in range(K)]

def generate_reward(arm):
    mean, var = reward_distributions[arm]
    return np.random.normal(mean, var)


def value_iteration_bandit(K, N, gamma=0.99, theta=1e-6):
    V = np.zeros(K)
    policy = np.zeros(K, dtype=int)
    Q = np.zeros((K, 1))

    for _ in range(N):
        for k in range(K):
            V[k] = np.max(Q[k])
            Q[k] = generate_reward(k) + gamma * V[k]
            policy[k] = np.argmax(Q[k])
    
    return policy, Q


def policy_iteration_bandit(K, N, gamma=0.99, theta=1e-6):
    policy = np.random.choice(K, size=1)
    V = np.zeros(K)
    Q = np.zeros((K, 1))

    for _ in range(N):
        for k in range(K):
            V[k] = generate_reward(k) + gamma * V[policy[0]]
            Q[k] = V[k]
        policy[0] = np.argmax(V)

    return policy, Q


def q_learning_bandit(K, N, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros(K)
    for _ in range(N):
        if random.uniform(0, 1) < epsilon:
            action = np.random.randint(K)
        else:
            action = np.argmax(Q)
        reward = generate_reward(action)
        Q[action] = Q[action] + alpha * (reward + gamma * np.max(Q) - Q[action])
    policy = np.argmax(Q)
    return policy, Q


class EpsilonGreedyAgent:
    def __init__(self, K, epsilon=0.1):
        self.K = K
        self.epsilon = epsilon
        self.Q = np.zeros(K)
        self.N = np.zeros(K)

    def select_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.K)
        else:
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class UCBAgent:
    def __init__(self, K, c=2):
        self.K = K
        self.c = c
        self.Q = np.zeros(K)
        self.N = np.zeros(K)
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
   
    policy_vi, Q_vi = value_iteration_bandit(K, N)
    print("Value Iteration - Optimal Policy:", policy_vi)
    print("Value Iteration - Q-Values:", Q_vi)

   
    policy_pi, Q_pi = policy_iteration_bandit(K, N)
    print("Policy Iteration - Optimal Policy:", policy_pi)
    print("Policy Iteration - Q-Values:", Q_pi)

   
    policy_ql, Q_ql = q_learning_bandit(K, N)
    print("Q-Learning - Optimal Policy:", policy_ql)
    print("Q-Learning - Q-Values:", Q_ql)

    
    agent_eg = EpsilonGreedyAgent(K)
    for t in range(N):
        action = agent_eg.select_action()
        reward = generate_reward(action)
        agent_eg.update(action, reward)
    print("Epsilon-Greedy - Estimated Q-values:", agent_eg.Q)

    
    agent_ucb = UCBAgent(K)
    for t in range(N):
        action = agent_ucb.select_action()
        reward = generate_reward(action)
        agent_ucb.update(action, reward)
    print("UCB - Estimated Q-values:", agent_ucb.Q)

if __name__ == "__main__":
    main()
