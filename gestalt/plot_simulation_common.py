import os
import json
import six
import numpy as np

from tree_distance import MRCADistanceMeasurer, MRCASpearmanMeasurer, RootRFDistanceMeasurer
from cell_lineage_tree import CellLineageTree

def get_true_model(file_name, tree_file_name, n_bcodes):
    with open(file_name, "rb") as f:
        true_model = six.moves.cPickle.load(f)
    with open(tree_file_name, "rb") as f:
        true_coll_tree = six.moves.cPickle.load(f)
    true_tree = true_model["true_subtree"]
    existing_strs = {}
    for node in true_tree:
        node.set_allele_list(
                node.allele_list.create_truncated_version(n_bcodes))
        node.sync_allele_events_list_str()
        if node.allele_events_list_str in existing_strs:
            count = existing_strs[node.allele_events_list_str]
            existing_strs[node.allele_events_list_str] += 1
            node.allele_events_list_str = "%s::%d" % (
                     node.allele_events_list_str,
                     count)
        else:
            existing_strs[node.allele_events_list_str] = 1

    return (true_model["true_model_params"], true_tree, true_coll_tree)

def get_result(res_file):
    with open(res_file, "rb") as f:
        result = six.moves.cPickle.load(f)
    for node in result.fitted_bifurc_tree:
        if node.abundance > 1:
            for idx in range(node.abundance):
                new_child = CellLineageTree(
                    node.allele_list,
                    node.allele_events_list,
                    node.cell_state,
                    dist = 0,
                    abundance = 1,
                    resolved_multifurcation = True)
                if idx > 0:
                    new_child.allele_events_list_str = "%s::%d" % (
                        new_child.allele_events_list_str,
                        idx)
                node.add_child(new_child)
    return (result.model_params_dict, result.fitted_bifurc_tree)

def get_target_lams(model_param_tuple):
    return model_param_tuple[0]["target_lams"]

def get_double_cut_weight(model_param_tuple):
    return model_param_tuple[0]["double_cut_weight"]
