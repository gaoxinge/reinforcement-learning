import math
import random


class ABTest:

    def __init__(self, arm_n):
        self.arm_n = arm_n

    def pull(self):
        return random.choice(range(self.arm_n))

    def update(self, arm, reward):
        pass
        
    def __str__(self):
        return 'ab test'
        
    def __repr__(self):
        return 'ABTest(arm_n)'.format(arm_n=self.arm_n)


class EpsilonGreedy:
    
    def __init__(self, arm_n, epsilon):
        self.arm_n = arm_n
        self.epsilon = epsilon
        self.values = [0.0 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]
        
    def pull(self):
        for arm in range(self.arm_n):
            if self.counts[arm] == 0:
                return arm
                
        if random.random() < self.epsilon:
            return random.choice(range(self.arm_n))
        else:
            m = max(self.values)
            return self.values.index(m)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        
    def __str__(self):
        return 'epsilon greedy: %s' % self.epsilon
        
    def __repr__(self):
        return 'EpsilonGreedy(arm_n, epsilon)'.format(arm_n=self.arm_n, epsilon=self.epsilon)


class Softmax:
    
    def __init__(self, arm_n, temperature):
        self.arm_n = arm_n
        self.temperature = temperature
        self.values = [0.0 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]
        
    def pull(self):
        for arm in range(self.arm_n):
            if self.counts[arm] == 0:
                return arm

        weights = [math.exp(v / self.temperature) for v in self.values]
        s = sum(weights)
        weights = [weight / s for weight in weights]
        r = random.choices(list(range(self.arm_n)), weights)[0]
        return r
        
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        
    def __str__(self):
        return 'softmax: %s' % self.temperature
        
    def __repr__(self):
        return 'Softmax(arm_n, temperature)'.format(arm_n=self.arm_n, temperature=self.temperature)


class UCB1:
    
    def __init__(self, arm_n):
        self.arm_n = arm_n
        self.values = [0.0 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]
        
    def pull(self):
        for arm in range(self.arm_n):
            if self.counts[arm] == 0:
                return arm
                
        t = sum(self.counts)
        ucbs = [self.values[arm] + math.sqrt((2 * math.log(t)) / self.counts[arm])
                for arm in range(self.arm_n)]
        m = max(ucbs)
        return ucbs.index(m)
        
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        
    def __str__(self):
        return 'ucb1'
        
    def __repr__(self):
        return 'UCB1(arm_n)'.format(arm_n=self.arm_n)


class TS:

    def __init__(self, arm_n):
        self.arm_n = arm_n
        self.alphas = [1 for _ in range(self.arm_n)]
        self.betas = [1 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]

    def pull(self):
        for arm in range(self.arm_n):
            if self.counts[arm] == 0:
                return arm

        thetas = [random.betavariate(alpha, beta) for alpha, beta in zip(self.alphas, self.betas)]
        m = max(thetas)
        return thetas.index(m)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward

    def __str__(self):
        return 'ts'

    def __repr(self):
        return 'TS(arm_n)'.format(arm_n=self.arm_n)
