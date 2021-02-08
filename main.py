import gym
import numpy as np
from collections import deque
from ddqn_agent import Agent


n_episodes = 10000

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    scores = deque(maxlen=10)

    for i in range(1, n_episodes+1):
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
