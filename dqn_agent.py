import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from replay_buffer import ReplayBuffer


BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
SYNC_TARGET_EVERY = 1000
LR = 1e-3
UPDATE_EVERY = 4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}\n")


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.policy_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

        self.eps = EPS_START
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0
        self.learn_count = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE, device)
            self.learn(experiences)

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            action_values = self.policy_network(state)
            return torch.argmax(action_values).item()

    def update_eps(self):
        self.eps = max(EPS_END, EPS_DECAY * self.eps)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_current = self.policy_network(states).gather(1, actions)

        Q_targets_next = self.target_network(next_states).max(1)[0].unsqueeze(1)
        Q_targets = rewards + GAMMA * Q_targets_next * (1 - dones)

        loss = F.mse_loss(Q_current, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % SYNC_TARGET_EVERY == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
