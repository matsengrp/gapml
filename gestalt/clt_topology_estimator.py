from typing import List, Dict, Tuple
import logging
import copy
import numpy as np

from cell_lineage_tree import CellLineageTree
from approximator import ApproximatorLB
from clt_estimator import CLTEstimator
from clt_likelihood_estimator import CLTPenalizedEstimator
from clt_likelihood_model import CLTLikelihoodModel

class CLTTopologyEstimator(CLTEstimator):
    def __init__(self,
            approximator: ApproximatorLB,
            bcode_meta,
            sess,
            multifurc_tree: CellLineageTree):
        self.approximator = approximator
        self.multifurc_tree = multifurc_tree
        self.bcode_meta = bcode_meta
        self.sess = sess

    def estimate(self, topology_iters:int=1, max_iters: int=1000):
        self.best_log_lik = -np.inf
        for i in range(topology_iters):
            print("working on multifurcation", i)
            is_done = self._resolve_multifurc(max_iters)
            if is_done:
                break
        logging.info("best one")
        logging.info(self.multifurc_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        return self.best_log_lik, self.multifurc_tree

    def _do_rearrangement(self, node, c_index):
        scratch_node = copy.deepcopy(node)
        node_str = node.allele_events_list_str
        # reconstruct the tree
        scratch_child = scratch_node.get_children()[c_index]
        assert scratch_child != node.get_children()[c_index]
        if scratch_child.allele_events_list_str == node_str:
            return None, None, None, None

        if node.up is not None:
            up_node = node.up
            up_node.add_child(scratch_node)
            up_node.remove_child(node)
            scratch_tree = self.multifurc_tree
        else:
            up_node = None
            scratch_tree = scratch_node
        new_inner_node = CellLineageTree(
                allele_list=node.allele_list,
                allele_events_list=node.allele_events_list,
                cell_state=node.cell_state)
        new_inner_node.add_feature("observed", node.observed)

        sisters = scratch_child.get_sisters()
        reattach_node = scratch_node
        for s in sisters:
            s.detach()
            new_inner_node.add_child(s)
        scratch_node.add_child(new_inner_node)
        logging.info(scratch_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        return scratch_tree, up_node, node, scratch_node

    def _resolve_multifurc(self, max_iters: int=1000):
        """
        @return whether or not there are more multifurcations to resolve
        """
        for node in self.multifurc_tree.traverse():
            children = node.get_children()
            children_strs = set([c.allele_events_list_str for c in children])
            node_str = node.allele_events_list_str
            num_children = len(children)
            if num_children > 2:
                # This is a multifurcation
                rearrangement_scores = -np.inf * np.ones(num_children)
                for c_i in range(num_children):
                    scratch_tree, orig_up, orig_node, scratch = self._do_rearrangement(node, c_i)
                    if scratch_tree is None:
                        continue
                    log_lik = self._estimate_likelihood(scratch_tree, max_iters)
                    logging.info("topology option SCORE %f", log_lik)
                    rearrangement_scores[c_i] = log_lik
                    if orig_up is not None:
                        orig_up.remove_child(scratch)
                        orig_up.add_child(orig_node)

                best_idx = np.argmax(rearrangement_scores)
                self.best_log_lik = rearrangement_scores[best_idx]
                self.multifurc_tree, _, _, _ = self._do_rearrangement(node, best_idx)
                # Stop with this node traversal and have the whole thing start again
                # Basically find a new multifurcation please.
                return False
        return True

    def _estimate_likelihood(self, tree, max_iters):
        num_nodes = len([t for t in tree.traverse()])
        target_lams = (0.3 * np.ones(self.bcode_meta.n_targets)
                    + np.random.uniform(size=self.bcode_meta.n_targets) * 0.08)

        res_model = CLTLikelihoodModel(
            tree,
            self.bcode_meta,
            self.sess,
            target_lams = target_lams,
            branch_len_inners = np.random.rand(num_nodes) * 0.1,
            cell_type_tree = None)
        estimator = CLTPenalizedEstimator(res_model, self.approximator, 0.2)
        pen_log_lik = estimator.fit(1, max_iters)
        return pen_log_lik
