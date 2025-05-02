"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

import numpy as np


class QLearning:
    def __init__(self, num_actions, features, policy_rng, learner_rng, alpha, eps):
        self.num_actions = num_actions
        self.features = features
        self.policy_rng = policy_rng
        self.learner_rng = learner_rng
        self.alpha = alpha # Step-Size
        self.eps = eps # The epsilon that decides explore/exploit

        # Randomly initialize parameters following standard normal
        self.parameters = learner_rng.randn(self.features.dim)

    def compute_qsa(self, state, action):
        feature = self.features(state, action)
        return self.parameters @ feature

    def get_action(self, curr_state, *args, **kwargs):
        """
        Samples action using epsilon-greedy strategy.
        """

        action = self.policy_rng.randint(self.num_actions)
        prob = self.policy_rng.uniform()

        # ========================================================
        # TODO: Implement epsilon-greedy strategy

        # Exploration case:
        if prob < self.eps:
            action = self.policy_rng.randint(self.num_actions)
        else: # Exploitation case:
            qvals = [self.compute_qsa(curr_state, a) for a in range(self.num_actions)] # Compute array of q(sk, a) for all actions a.
            action = np.argmax(qvals) # Argmax that sheisce

        return int(action)

        # ========================================================



    def compute_update(self, curr_state, action, reward, done, next_state):
        """
        Computes the update.

        Recall that the update rule is:
            (r + gamma * max_b' Q_w(s', a') - Q_w(s, a)) * grad_w Q_w(s, a)

        NOTE: Upon reaching terminal, we no longer bootstrap from the next Q-value.
        """

        update = None
        gamma = 0.1
        # ========================================================
        # TODO: Implement the update rule

        # Compute the current Q-value for (curr_state, action)
        feature = self.features(curr_state, action) # phi(s, a)
        current_q = self.parameters @ feature # Q(s, a) for this phi = wTphi

        ## NOTE: We multiply with alpha in learn() and update wk there as well.

        # If terminal, target is just the reward
        if done:
            target = reward
        else:
            # Otherwise, target = reward + gamma * max_{a'} Q(s', a')
            qnext = [self.compute_qsa(next_state, a) for a in range(self.num_actions)]
            target = reward + gamma * np.max(qnext)

        # TD error: (target - current Q-value)
        td_error = target - current_q

        # For a linear function approximator, grad_w Q = feature
        update = td_error * feature
        # ========================================================

        return update

    def learn(self, curr_state, action, reward, done, next_state, *args, **kwargs):
        """
        Updates the parameters of the Q-function
        """

        update = self.compute_update(curr_state, action, reward, done, next_state)
        self.parameters += self.alpha * update