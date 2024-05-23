import gym
import numpy as np

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
env.reset()

# Value Iteration function with modified variable names
def value_iteration(env, gamma=0.99, theta=1e-9):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    values = np.zeros(num_states)
    best_policy = np.zeros(num_states, dtype=int)

    state_changed = True
    while state_changed:
        state_changed = False
        for state in range(num_states):
            action_values = []
            for action in range(num_actions):
                value = 0
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    value += prob * (reward + gamma * values[next_state])
                action_values.append(value)
            best_action = np.argmax(action_values)
            best_action_value = action_values[best_action]
            if np.abs(values[state] - best_action_value) > theta:
                values[state] = best_action_value
                best_policy[state] = best_action
                state_changed = True

    return best_policy, values

# Policy Iteration function with modified variable names
def policy_iteration(env, gamma=0.99):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    current_policy = np.random.choice(num_actions, size=num_states)
    values = np.zeros(num_states)
    
    is_policy_stable = False
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(num_states):
                value = 0
                action = current_policy[state]
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    value += prob * (reward + gamma * values[next_state])
                delta = max(delta, np.abs(value - values[state]))
                values[state] = value
            if delta < 1e-10:
                break
        
        # Policy Improvement
        is_policy_stable = True
        for state in range(num_states):
            old_action = current_policy[state]
            action_values = []
            for action in range(num_actions):
                value = 0
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    value += prob * (reward + gamma * values[next_state])
                action_values.append(value)
            new_action = np.argmax(action_values)
            current_policy[state] = new_action
            if new_action != old_action:
                is_policy_stable = False

    return current_policy, values

# Q-Learning function with modified variable names
def q_learning(env, num_episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_values = np.zeros((num_states, num_actions))

    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(q_values[state])
            next_state, reward, done, _, _ = env.step(action)
            td_target = reward + gamma * np.max(q_values[next_state])
            td_error = td_target - q_values[state, action]
            q_values[state, action] += alpha * td_error
            state = next_state

    best_policy = np.argmax(q_values, axis=1)
    return best_policy, q_values

# Helper function to print the policy
def print_policy_with_symbols(policy, grid_size):
    symbols = {
        0: 'L', 1: 'D', 2: 'R', 3: 'U'
    }
    policy_grid = np.array([symbols[action] for action in policy]).reshape(grid_size)
    for row in policy_grid:
        print(' '.join(row))

# Run the algorithms and display the policies
grid_size = (4, 4)  # for FrozenLake-v1 4x4
best_policy_vi, _ = value_iteration(env)
best_policy_pi, _ = policy_iteration(env)
best_policy_ql, _ = q_learning(env)

print("Value Iteration Best Policy:")
print_policy_with_symbols(best_policy_vi, grid_size)

print("Policy Iteration Best Policy:")
print_policy_with_symbols(best_policy_pi, grid_size)

print("Q-Learning Best Policy:")
print_policy_with_symbols(best_policy_ql, grid_size)
