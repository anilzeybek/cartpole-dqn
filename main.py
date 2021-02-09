import gym
import numpy as np
import sys
from collections import deque


N_EPISODES = 10000
env = gym.make('LunarLander-v2')


if sys.argv[1] == "dqn":
    from dqn_agent import DQNAgent

    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    print("Using DQN algorithm")
elif sys.argv[1] == "ddqn":
    from ddqn_agent import DDQNAgent

    agent = DDQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    print("Using DDQN algorithm")
else:
    from dueling_agent import DuelingAgent

    agent = DuelingAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    print("Using Dueling algorithm")


scores = deque(maxlen=10)
for i in range(1, N_EPISODES+1):
    state = env.reset()
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

    scores.append(score)
    agent.update_eps()

    mean_scores = np.mean(scores)
    if mean_scores > 250:
        print("Solved!")
        break

    print(f'\rEpisode: {i}\tAverage Score: {mean_scores:.2f}', end="")
    if i % 10 == 0:
        print(f'\rEpisode: {i}\tAverage Score: {mean_scores:.2f}')

while True:
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        env.render()
