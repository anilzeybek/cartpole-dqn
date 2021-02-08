import gym
import numpy as np
from collections import deque
from dqn_agent import Agent


n_episodes = 2000

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    scores = deque(maxlen=100)

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

        print('\rEpisode: {}\tAverage Score: {:.2f}'.format(i, np.mean(scores)), end="")
        if i % 100 == 0:
            print('\rEpisode: {}\tAverage Score: {:.2f}'.format(i, np.mean(scores)))
