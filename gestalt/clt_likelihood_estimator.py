import time
from tensorflow import Session
import numpy as np
import scipy.linalg
from typing import List, Tuple
from numpy import ndarray

from clt_estimator import CLTEstimator
from cell_lineage_tree import CellLineageTree
from clt_likelihood_model import CLTLikelihoodModel
import ancestral_events_finder as anc_evt_finder
from approximator import ApproximatorLB
from transition_matrix import TransitionMatrixWrapper, TransitionMatrix
from indel_sets import TargetTract

from state_sum import StateSum
from common import target_tract_repr_diff

class CLTLassoEstimator(CLTEstimator):
    """
    Likelihood estimator

    TODO: Right now this ignores cell type. we'll add it in later
    """
    def __init__(
        self,
        sess: Session,
        penalty_param: float,
        model: CLTLikelihoodModel,
        approximator: ApproximatorLB):
        """
        @param penalty_param: lasso penalty parameter
        @param model: initial CLT model params
        """
        self.sess = sess
        self.penalty_param = penalty_param
        self.model = model
        self.approximator = approximator

        # Annotate with ancestral states
        anc_evt_finder.annotate_ancestral_states(model.topology, model.bcode_meta)
        # Create the skeletons for the transition matrices -- via state sum approximation
        self.transition_mat_wrappers = self.approximator.create_transition_matrix_wrappers(model.topology)

    def get_likelihood(self, model: CLTLikelihoodModel, get_grad: bool = False):
        """
        Does the Felsenstein algo to efficiently calculate the likelihood of the tree,
        assuming state_sum contains all the possible ancestral states (though that's not
        actually true. it's an approximation)

        @return The likelihood for proposed theta, the gradient too if requested
        """
        transition_matrices = model.initialize_transition_matrices(self.transition_mat_wrappers)
        # TODO: Initialize trim probs -- just like we did for transition matrices!

        log_lik = 0
        L = dict() # Stores normalized probs
        pt_matrix = dict()
        trim_probs = dict()
        for node in model.topology.traverse("postorder"):
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
                    trim_probs[child.node_id] = self._get_trim_probs(
                            model,
                            ch_trans_mat,
                            node,
                            child)

                    # Create the probability matrix exp(Qt) = A * exp(Dt) * A^-1
                    branch_len = model.branch_lens[child.node_id].eval()
                    pt_matrix[child.node_id] = np.dot(
                            ch_trans_mat.A,
                            np.dot(
                                np.diag(np.exp(ch_trans_mat.D * branch_len)),
                                ch_trans_mat.A_inv))

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
                        node_trans_mat = transition_matrices[node.node_id]
                        down_probs = self._reorder_likelihoods(
                            ch_ordered_down_probs,
                            node.state_sum.tts_list,
                            node_trans_mat,
                            ch_trans_mat)

                        L[node.node_id] *= down_probs
                        if down_probs.max() == 0:
                            raise ValueError("Why is everything zero?")
                    else:
                        # For the root node, we just want the probability where the root node is unmodified
                        # No need to reorder
                        ch_id = ch_trans_mat.key_dict[()]
                        L[node.node_id] *= ch_ordered_down_probs[ch_id]

                scaler = L[node.node_id].max()
                if scaler == 0:
                    raise ValueError("Why is everything zero?")
                L[node.node_id] /= scaler
                log_lik += np.sum(np.log(scaler))

        log_lik += np.log(L[model.root_node_id])
        return log_lik

    def _get_trim_probs(self,
            model: CLTLikelihoodModel,
            ch_trans_mat: TransitionMatrix,
            node: CellLineageTree,
            child: CellLineageTree):
        """
        @param model: model parameter
        @param ch_trans_mat: the transition matrix corresponding to child node (we make sure the entries in the trim prob matrix match
                        the order in ch_trans_mat)
        @param node: the parent node
        @param child: the child node

        @return matrix of conditional probabilities of each trim
        """
        trim_prob_mat = np.ones((ch_trans_mat.num_states, ch_trans_mat.num_states))

        for node_tts in node.state_sum.tts_list:
            node_tts_key = ch_trans_mat.key_dict[node_tts]
            for child_tts in child.state_sum.tts_list:
                child_tts_key = ch_trans_mat.key_dict[child_tts]
                diff_target_tracts = target_tract_repr_diff(node_tts, child_tts)
                trim_prob = model.get_prob_unmasked_trims(child.anc_state, diff_target_tracts)
                trim_prob_mat[node_tts_key, child_tts_key] = trim_prob

        return trim_prob_mat

    def _reorder_likelihoods(self,
            vec_lik: ndarray,
            tts_list: List[Tuple[TargetTract]],
            node_trans_mat: TransitionMatrixWrapper,
            ch_trans_mat: TransitionMatrixWrapper):
        """
        @param vec_lik: the thing to be re-ordered
        @param tts_list: list of target tract reprs to include in the vector
                        rest can be set to zero
        @param node_trans_mat: provides the desired ordering
        @param ch_trans_mat: provides the ordering used in vec_lik

        @return the reordered version of vec_lik according to the order in node_trans_mat
        """
        down_probs = np.zeros((node_trans_mat.num_states, 1))
        for tts in tts_list:
            ch_tts_id = ch_trans_mat.key_dict[tts]
            node_tts_id = node_trans_mat.key_dict[tts]
            down_probs[node_tts_id] = vec_lik[ch_tts_id]
        return down_probs
