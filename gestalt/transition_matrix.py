from typing import Dict
import numpy as np

class TransitionMatrixWrapper:
    def __init__(self, matrix_dict: Dict[str, Dict[str, float]]):
        """
        Create transition matrix given the dictionary representation.
        Assume each key in the dictionary corresponds to a possible
        target tract representation state.
        """
        # Number each key
        self.key_list = []
        self.key_dict = dict()
        i = 0
        for key in matrix_dict.keys():
            self.key_list.append(key)
            self.key_dict[key] = i
            i += 1
        self.num_states = i

        # Create matrix
        self.matrix = np.zeros((self.num_states, self.num_states))
        for i, key in enumerate(self.key_list):
            for to_key, val in matrix_dict[key].items():
                self.matrix[i, self.key_dict[to_key]] = val

        # Store the matrix diagonlization decomposition
        self.D, self.A = np.linalg.eig(self.matrix)
        self.A_inv = np.linalg.inv(self.A)

    def __str__(self):
        return "Key list: %s \n Matrix: %s" % (self.key_list, self.matrix)
