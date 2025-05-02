import copy
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

MAP = [
    "FFFFFFFF",
    "FFFFFFFF",
    "FFGHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFSFF",
]


class GridWorld:
    def __init__(self):
        self.slip_prob = [1.0 / 3.0] * 3
        desc = MAP

        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 5
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self._P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            curr_letter = desc[row, col]
            terminated = False
            reward = float(curr_letter == b"G") - float(curr_letter == b"H")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    li = self._P[s][a]
                    letter = desc[row, col]
                    if self.slip_prob[1] < 1 and a != STAY:
                        for slip_dir, b in enumerate([(a - 1) % 4, a, (a + 1) % 4]):
                            li.append(
                                (
                                    self.slip_prob[slip_dir],
                                    *update_probability_matrix(row, col, b),
                                )
                            )
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        self.n_s = nS

    @property
    def transition(self):
        return copy.deepcopy(self._P)
