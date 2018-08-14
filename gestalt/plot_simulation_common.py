import os
import json
import six
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tree_distance import MRCADistanceMeasurer, MRCASpearmanMeasurer, RootRFDistanceMeasurer
from cell_lineage_tree import CellLineageTree

def get_true_model(file_name, tree_file_name, n_bcodes):
    """
    Read true model
    """
    with open(file_name, "rb") as f:
        true_model = six.moves.cPickle.load(f)
    if tree_file_name is not None:
        with open(tree_file_name, "rb") as f:
            true_coll_tree = six.moves.cPickle.load(f)
    else:
        true_coll_tree = None

    # Rename nodes if they have the same first few barcodes
    true_tree = true_model["true_subtree"]
    existing_strs = {}
    for node in true_tree:
        node.set_allele_list(
                node.allele_list.create_truncated_version(n_bcodes))
        node.sync_allele_events_list_str()
        if node.allele_events_list_str in existing_strs:
            count = existing_strs[node.allele_events_list_str]
            existing_strs[node.allele_events_list_str] += 1
            node.allele_events_list_str = "%s==%d" % (
                     node.allele_events_list_str,
                     count)
        else:
            existing_strs[node.allele_events_list_str] = 1
    return (true_model["true_model_params"], true_tree, true_coll_tree)

def get_result(res_file):
    """
    Read fitted model
    """
    with open(res_file, "rb") as f:
        result = six.moves.cPickle.load(f)
    raw_tree = result.fitted_bifurc_tree.copy()

    # Create appropriate number of leaves to match abundance
    for node in result.fitted_bifurc_tree:
        curr_node = node
        for idx in range(node.abundance - 1):
            new_child = CellLineageTree(
                curr_node.allele_list,
                curr_node.allele_events_list,
                curr_node.cell_state,
                dist = 0,
                abundance = 1,
                resolved_multifurcation = True)
            new_child.allele_events_list_str = "%s==%d" % (
                new_child.allele_events_list_str,
                idx + 1)
            copy_leaf = CellLineageTree(
                curr_node.allele_list,
                curr_node.allele_events_list,
                curr_node.cell_state,
                dist = 0,
                abundance = 1,
                resolved_multifurcation = True)
            copy_leaf.allele_events_list_str = curr_node.allele_events_list_str
            curr_node.add_child(new_child)
            curr_node.add_child(copy_leaf)
            curr_node = new_child
    return (result.model_params_dict, result.fitted_bifurc_tree, raw_tree)

def get_rand_tree(res_file):
    """
    Reads a tree from parsimony -- no fitted model params since this is just from parsimony
    """
    with open(res_file, "rb") as f:
        parsimony_tree_dict = six.moves.cPickle.load(f)
    parsimony_tree = parsimony_tree_dict["tree"]

    # Create appropriate number of leaves to match abundance
    for node in parsimony_tree:
        curr_node = node
        for idx in range(node.abundance - 1):
            new_child = CellLineageTree(
                curr_node.allele_list,
                curr_node.allele_events_list,
                curr_node.cell_state,
                dist = 0,
                abundance = 1,
                resolved_multifurcation = True)
            new_child.allele_events_list_str = "%s==%d" % (
                new_child.allele_events_list_str,
                idx + 1)
            copy_leaf = CellLineageTree(
                curr_node.allele_list,
                curr_node.allele_events_list,
                curr_node.cell_state,
                dist = 0,
                abundance = 1,
                resolved_multifurcation = True)
            copy_leaf.allele_events_list_str = curr_node.allele_events_list_str
            curr_node.add_child(new_child)
            curr_node.add_child(copy_leaf)
            curr_node = new_child
    return (None, parsimony_tree)

def get_target_lams(model_param_tuple):
    target_lams = model_param_tuple[0]["target_lams"]
    double_weight = model_param_tuple[0]["double_cut_weight"]
    trim_long = model_param_tuple[0]["trim_long_factor"]
    return np.concatenate([target_lams, double_weight, trim_long])

def get_only_target_lams(model_param_tuple):
    return model_param_tuple[0]["target_lams"]

def get_double_cut(model_param_tuple):
    return model_param_tuple[0]["double_cut_weight"]

def print_results(settings, n_bcode_results, n_bcode, print_keys):
    settings = [str(s) for s in settings]
    for idx, setting in enumerate(settings):
        #print(n_bcode_results["tau"][idx])
        #print(n_bcode_results["seeds"][idx])
        size = len(n_bcode_results["mrca"][idx])
        print_list = [
                setting,
                size,
                np.mean(n_bcode_results["leaves"][idx]),
                np.sqrt(np.var(n_bcode_results["leaves"][idx])/size)]
        print_template = "%s & %d & %.01f (%.01f)"
        for key in print_keys:
            size = len(n_bcode_results[key][idx])
            print_list.append(np.mean(n_bcode_results[key][idx]))
            print_list.append(np.sqrt(np.var(n_bcode_results["mrca"][idx])/size))
            print_template += "& %.03f (%.03f)"
        print_template += "\\\\"
        print(print_template % tuple(print_list))

def plot_mrca_matrix(mrca_mat, file_name: str, tot_time: float = 1):
    plt.clf()
    plt.imshow(mrca_mat, vmin=0, vmax=tot_time)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.savefig(file_name)

