import numpy as np

class MultiArmedBanditEnvironment:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.probabilities = np.random.rand(num_arms)  # Random probabilities for each arm
        self.best_arm_index = np.argmax(self.probabilities)

    def reset(self):
        return 0  # single state

    def step(self, action):
        reward = np.random.binomial(1, self.probabilities[action])
        return 0, reward, False, {}

    def get_optimal_arm(self):
        return self.best_arm_index, self.probabilities[self.best_arm_index]

class GreedyAgent:
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.arm_counts = np.zeros(num_arms)
        self.arm_values = np.zeros(num_arms)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.arm_values)

    def update(self, action, reward):
        self.arm_counts[action] += 1
        self.arm_values[action] += (reward - self.arm_values[action]) / self.arm_counts[action]

    def run(self, env, steps):
        total_reward = 0
        for _ in range(steps):
            action = self.select_action()
            _, reward, _, _ = env.step(action)
            self.update(action, reward)
            total_reward += reward
        return total_reward

class UCBExplorer:
    def __init__(self, num_arms, c=2):
        self.num_arms = num_arms
        self.c = c
        self.arm_counts = np.zeros(num_arms)
        self.arm_values = np.zeros(num_arms)
        self.total_counts = 0

    def select_action(self):
        ucb_values = self.arm_values + self.c * np.sqrt(np.log(self.total_counts + 1) / (self.arm_counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.arm_counts[action] += 1
        self.total_counts += 1
        self.arm_values[action] += (reward - self.arm_values[action]) / self.arm_counts[action]

    def run(self, env, steps):
        total_reward = 0
        for _ in range(steps):
            action = self.select_action()
            _, reward, _, _ = env.step(action)
            self.update(action, reward)
            total_reward += reward
        return total_reward

num_arms = 10  # Number of arms
steps = 1000  # Number of steps

env = MultiArmedBanditEnvironment(num_arms)

# Greedy Agent
greedy_agent = GreedyAgent(num_arms, epsilon=0.1)
greedy_reward = greedy_agent.run(env, steps)

# UCB Agent
ucb_explorer = UCBExplorer(num_arms, c=2)
ucb_reward = ucb_explorer.run(env, steps)

print(f"Greedy Agent Total Reward: {greedy_reward}")
print(f"UCB Explorer Total Reward: {ucb_reward}")

# Optimal strategy
optimal_arm_index, optimal_reward = env.get_optimal_arm()
print(f"Optimal Arm Index: {optimal_arm_index}, Optimal Reward Rate: {optimal_reward * steps}")
