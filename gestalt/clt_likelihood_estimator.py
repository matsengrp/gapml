import time
import numpy as np
from typing import List
from numpy import ndarray

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
import ancestral_events_finder as anc_evt_finder
from approximator import ApproximatorLB

from state_sum import StateSum
from common import target_tract_repr_diff

class CLTCalculations:
    """
    Stores parameters useful for likelihood/gradient calculations
    """
    def __init__(self, dl_dbranch_lens: ndarray, dl_dtarget_lams: ndarray, dl_dcell_type_lams: ndarray):
        self.dl_dbranch_lens = dl_dbranch_lens
        self.dl_dtarget_lams = dl_dtarget_lams
        self.dl_dcell_type_lams = dl_dcell_type_lams

class CLTLassoEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    def __init__(
        self,
        penalty_param: float,
        model_params: CLTLikelihoodModel,
        approximator: ApproximatorLB):
        """
        @param penalty_param: lasso penalty parameter
        @param model_params: initial CLT model params
        """
        self.penalty_param = penalty_param
        self.model_params = model_params
        self.approximator = approximator

        # Annotate with ancestral states
        anc_evt_finder.annotate_ancestral_states(model_params.topology, model_params.bcode_meta)
        # Construct transition boolean matrix -- via state sum approximation
        self.approximator.annotate_state_sum_transitions(model_params.topology)


    def get_likelihood(self, model_params: CLTLikelihoodModel, get_grad: bool = False):
        """
        Does the Felsenstein algo to efficiently calculate the likelihood of the tree,
        assuming state_sum contains all the possible ancestral states (though that's not
        actually true. it's an approximation)

        @return The likelihood for proposed theta, the gradient too if requested
        """
        transition_matrices = model_params.create_transition_matrices()

        L = dict()
        pt_matrix = dict()
        trim_probs = dict()
        for node in model_params.topology.traverse("postorder"):
            if node.is_leaf():
                node_trans_mat = transition_matrices[node.node_id]
                L[node.node_id] = np.zeros((node_trans_mat.num_states, 1))
                assert len(node.state_sum.tts_list) == 1
                tts_key = node_trans_mat.key_dict[node.state_sum.tts_list[0]]
                L[node.node_id][tts_key] = 1
            else:
                L[node.node_id] = 1
                for child in node.children:
                    ch_trans_mat = transition_matrices[child.node_id]

                    # Get the trim probabilities
                    trim_probs[child.node_id] = np.ones((ch_trans_mat.num_states, ch_trans_mat.num_states))
                    for node_tts in node.state_sum.tts_list:
                        node_tts_key = ch_trans_mat.key_dict[node_tts]
                        for child_tts in child.state_sum.tts_list:
                            child_tts_key = ch_trans_mat.key_dict[child_tts]
                            diff_target_tracts = target_tract_repr_diff(node_tts, child_tts)
                            trim_prob = model_params.get_prob_unmasked_trims(child.anc_state, diff_target_tracts)
                            trim_probs[child.node_id][node_tts_key, child_tts_key] = trim_prob

                    # Create the probability matrix exp(Qt) = A * exp(Dt) * A^-1
                    branch_len = model_params.branch_lens[child.node_id]
                    pt_matrix[child.node_id] = np.dot(ch_trans_mat.A, np.dot(np.diag(np.exp(ch_trans_mat.D * branch_len)), ch_trans_mat.A_inv))

                    # Get the probability for the data descended from the child node, assuming that the node
                    # has a particular target tract repr.
                    # These down probs are ordered according to the child node's numbering of the TTs states
                    ch_ordered_down_probs = np.dot(
                        np.multiply(
                            pt_matrix[child.node_id],
                            trim_probs[child.node_id]),
                        L[child.node_id])

                    if not node.is_root():
                        # Reorder summands according to node's numbering of tts states
                        # TODO: refactor this to be in another function so I can test this guy
                        node_trans_mat = transition_matrices[node.node_id]
                        down_probs = np.zeros((node_trans_mat.num_states, 1))
                        for tts in node.state_sum.tts_list:
                            ch_id = ch_trans_mat.key_dict[tts]
                            node_id = node_trans_mat.key_dict[tts]
                            down_probs[node_id] = ch_ordered_down_probs[ch_id]

                        L[node.node_id] *= down_probs
                    else:
                        # For the root node, we just want the probability where the root node is unmodified
                        # No need to reorder
                        ch_id = ch_trans_mat.key_dict[()]
                        L[node.node_id] *= ch_ordered_down_probs[ch_id]

        print("lik root", L[model_params.root_node_id])
        1/0
        return lik_root
