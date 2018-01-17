from typing import Dict
import scipy.sparse as sparse
import numpy as np

class TransitionMatrixWrapper:
    def __init__(self, matrix_dict: Dict[str, Dict[str, float]]):
        """
        Create sparse matrix matrix given the dictionary representation
        Assume each key in the dictionary corresponds to a matrix row

        TODO: decide if we should still call this "sparse matrix"
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
        #coo_rows = []
        #coo_cols = []
        #coo_vals = []
        #for i, key in enumerate(self.key_list):
        #    for to_key, val in matrix_dict[key].items():
        #        coo_rows.append(i)
        #        coo_cols.append(self.key_dict[to_key])
        #        coo_vals.append(val)

        #coo_rows.append(0)
        #coo_cols.append(0)
        #coo_vals.append(1)

        #coo_rows.append(1)
        #coo_cols.append(1)
        #coo_vals.append(1)

        #sparse_matrix = sparse.coo_matrix((coo_vals, (coo_rows, coo_cols)))
        #self.matrix = sparse_matrix.tocsr()
        #self.diag_mat, self.right_eig_mat = sparse.linalg.eigs(self.matrix, k=self.num_states)
        self.diag_mat, self.right_eig_mat = np.linalg.eig(self.matrix)
        self.right_eig_mat_inv = np.linalg.inv(self.right_eig_mat)

    def __str__(self):
        return "Key list: %s \n Matrix: %s" % (self.key_list, self.matrix)
