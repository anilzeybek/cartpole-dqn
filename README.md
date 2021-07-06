Solving lunarlander environment from gym with dqn, double dqn, dueling network architectures and prioritized experience replay with proportional prioritization algorithms.

Note that PER algorithm works slower than others because of not using some advanced data structures to decrease complexity.

Dependencies:
- Gym
- PyTorch
- Numpy

Usage: `python3 main.py [dqn, ddqn, per, duel]`

Papers of each of these algorithms:
- https://www.nature.com/articles/nature14236
- https://arxiv.org/abs/1509.06461
- https://arxiv.org/abs/1511.06581
- https://arxiv.org/abs/1511.05952

