import gym
import numpy as np
import sys
from collections import deque


N_EPISODES = 10000
env = gym.make('LunarLander-v2')

from policy_gradient_agent import PolicyGradientAgent

agent = PolicyGradientAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
print("Using REINFORCE algorithm")

scores = deque(maxlen=10)
for i in range(1, N_EPISODES+1):
    state = env.reset()
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.store_reward(reward)
        state = next_state
        score += reward

    agent.learn()
    scores.append(score)

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
