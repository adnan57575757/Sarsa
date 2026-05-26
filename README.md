# SARSA 📘

SARSA (State–Action–Reward–State–Action) is an **on-policy reinforcement learning algorithm**. Unlike Q-Learning, which is off-policy, SARSA updates its action-value function based on the action actually taken by the agent following its current policy.

---

## 🔑 Key Concepts
- **State (s):** Current situation of the agent.
- **Action (a):** Choice available to the agent.
- **Reward (r):** Feedback received after taking an action.
- **Next State (s'):** Resulting state after the action.
- **Next Action (a'):** Action chosen in the next state using the same policy.
- **Q-value (Q(s,a)):** Expected future reward of taking action `a` in state `s`.

---

## ⚙️ Algorithm
1. Initialize Q-table with zeros.
2. For each episode:
   - Start from an initial state.
   - Choose an action using ε-greedy.
   - Perform the action, observe reward and next state.
   - Choose next action using ε-greedy.
   - Update Q-value:

   

\[
   Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma Q(s',a') - Q(s,a) \Big]
   \]



   where:
   - \(\alpha\) = learning rate  
   - \(\gamma\) = discount factor  
   - \(s'\) = next state  
   - \(a'\) = next action  

3. Repeat until convergence.

---

## 🧩 Example (Python)
```python
import numpy as np
import gym

env = gym.make("CliffWalking-v0")
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1   # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1 # exploration rate
episodes = 500

for episode in range(episodes):
    state = env.reset()[0]
    action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
    done = False
    
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[next_state])
        
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        
        state, action = next_state, next_action
