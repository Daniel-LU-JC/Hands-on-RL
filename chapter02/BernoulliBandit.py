import numpy as np


class BernoulliBandit:
    """ multi-armed bandit, MAB; input param is the number of arms """
    def __init__(self, arm):
        self.probs = np.random.uniform(size=arm)  # probability for each arm
        self.best_idx = np.argmax(self.probs)  # arm with the highest probability
        self.best_prob = self.probs[self.best_idx]  # highest probability
        self.K = arm

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)
bandit_10_arm = BernoulliBandit(10)
print("of 10 arms, the highest probability is %.4f at arm %d" %
      (bandit_10_arm.best_prob, bandit_10_arm.best_idx))


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # times of attempts at each arm
        self.regret = 0  # cumulative regret
        self.actions = []  # list of actions
        self.regrets = []  # list of cumulative regrets

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError  # to be implemented for various strategies

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()  # make a decision
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
