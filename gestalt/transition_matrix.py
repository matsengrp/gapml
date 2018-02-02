from typing import Dict, Tuple
import numpy as np

class TransitionMatrixWrapper:
    def __init__(self, matrix_dict: Dict):
        """
        This is a skeleton of a transition matrix
        matrix_dict's key is the start target tract repr.
        matrix_dict's value is another dictionary that maps the end target tract repr to its TargetTract event
        """
        self.matrix_dict = matrix_dict

        # Number each key
        self.key_list = []
        self.key_dict = dict()
        i = 0
        for key in matrix_dict.keys():
            self.key_list.append(key)
            self.key_dict[key] = i
            i += 1
        self.num_states = i

class TransitionMatrix:
    def __init__(self,
            matrix_dict: Dict[str, Dict[str, float]],
            matrix_grad_dict: Dict[str, Dict[str, Tuple]] = None):
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

        # Store the gradient of the instantaneous transition matrix too
        if matrix_grad_dict is not None and False:
            self.grad_matrices = []
            key0 = self.key_list[0]
            vals = list(matrix_grad_dict[key0].values())
            self.num_p = vals[0].size
            for j in range(self.num_p):
                grad_matrix = np.zeros((self.num_states, self.num_states))
                for i, key in enumerate(self.key_list):
                    for to_key, val in matrix_grad_dict[key].items():
                        grad_matrix[i, self.key_dict[to_key]] = val[j]
                self.grad_matrices.append(grad_matrix)

    def __str__(self):
        return "Key list: %s \n Matrix: %s" % (self.key_list, self.matrix)
