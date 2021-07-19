import numpy as np
import torch
import torch.nn.functional as F
from reinforce_model import PolicyNetwork
import torch.optim as optim


LR = 0.0005
GAMMA = 0.99


class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        probabilities = F.softmax(self.policy(state))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        G = np.zeros_like(self.reward_memory)

        # in below nested loop, we calculate returns for all visited states during the episode
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= GAMMA

            G[t] = G_sum

        G = torch.tensor(G, dtype=torch.float)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss -= g * logprob  # -= because we want gradient ascent, not descent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
