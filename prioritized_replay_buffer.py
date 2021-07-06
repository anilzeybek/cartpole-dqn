from collections import namedtuple, deque
import numpy as np
import random
import torch


class Experience:
    def __init__(self, state, action, reward, next_state, done, priority):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.priority = priority

    def update_priority(self, new_priority):
        self.priority = new_priority


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha):
        self.memory = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.priorities = []

    def store_transition(self, state, action, reward, next_state, done, priority):
        e = Experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def update_probabilities(self):
        sum_priorities = sum(e.priority**self.alpha for e in self.memory)
        self.probabilities = [e.priority**self.alpha / sum_priorities for e in self.memory]

    def sample(self, batch_size, device):
        # TODO: here is problematic right now, a better implementation needed because
        # memory grows faster than probabilities list, which is error
        experience_indexes = np.random.choice(len(self.memory), batch_size, p=self.probabilities)
        experiences = [self.memory[i] for i in experience_indexes]

        return experiences

    def __len__(self):
        return len(self.memory)
