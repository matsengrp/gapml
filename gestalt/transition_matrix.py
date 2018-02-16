from typing import Dict, Tuple, List
import numpy as np

from indel_sets import TargetTract

class TransitionMatrixWrapper:
    def __init__(self, matrix_dict: Dict, key_list: List[Tuple[TargetTract]] = None):
        """
        This is a skeleton of a transition matrix
        matrix_dict's key is the start target tract repr.
        matrix_dict's value is another dictionary that maps the end target tract repr to its TargetTract event

        assumes that every value in the matrix dictionary shows up as a key in the dictionary
        TODO: add an assertion for this assumption!
        """
        self.matrix_dict = matrix_dict

        # Number each key
        self.key_list = [k for k in matrix_dict.keys()] if key_list is None else key_list
        assert(len(self.key_list) == len(matrix_dict.keys()))
        self.key_dict = {
            key: int(i) for i, key in enumerate(self.key_list)}
        self.num_likely_states = len(self.key_list)
