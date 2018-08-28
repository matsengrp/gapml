import os
import json
import six
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tree_distance import *
from cell_lineage_tree import CellLineageTree
from common import assign_rand_tree_lengths
#from plot_mrca_matrices import plot_tree

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

def _get_leaved_result(bifurc_tree):
    """
    Create appropriate number of leaves to match abundance
    Just attach with zero distance to the existing leaf node
    """
    leaved_tree = bifurc_tree.copy()
    for node in leaved_tree:
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
    return leaved_tree

def get_result(res_file):
    """
    Read fitted model
    """
    with open(res_file, "rb") as f:
        result = six.moves.cPickle.load(f)["refit"]

    # Create appropriate number of leaves to match abundance
    leaved_bifurc_tree = _get_leaved_result(result.fitted_bifurc_tree)
    return (result.model_params_dict, leaved_bifurc_tree, result.fitted_bifurc_tree)

def get_rand_tree(res_file):
    """
    Reads a tree from parsimony -- no fitted model params since this is just from parsimony
    """
    with open(res_file, "rb") as f:
        parsimony_tree_dict = six.moves.cPickle.load(f)
    parsimony_tree = parsimony_tree_dict["tree"]
    raw_pars_tree = parsimony_tree.copy()

    # Create appropriate number of leaves to match abundance
    for node in parsimony_tree:
        curr_node = node
        if node.abundance <= 1:
            continue
        for idx in range(node.abundance):
            new_child = CellLineageTree(
                node.allele_list,
                node.allele_events_list,
                node.cell_state,
                dist = 0,
                abundance = 1,
                resolved_multifurcation = True)
            if idx > 0:
                new_child.allele_events_list_str = "%s==%d" % (
                    new_child.allele_events_list_str,
                    idx)
            curr_node.add_child(new_child)
        curr_node.abundance = node.abundance
    return (None, parsimony_tree, raw_pars_tree)

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
        size = len(n_bcode_results[print_keys[0]][idx])
        print_list = [
                setting,
                size,
                np.mean(n_bcode_results["leaves"][idx]),
                np.sqrt(np.var(n_bcode_results["leaves"][idx])/size)]
        print_template = "%s & %d & %.01f (%.01f)"
        for key in print_keys:
            size = len(n_bcode_results[key][idx])
            #print(key, idx, n_bcode_results[key][idx])
            print_list.append(np.mean(n_bcode_results[key][idx]))
            print_list.append(np.sqrt(np.var(n_bcode_results[key][idx])/size))
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

def plot_tree(
        tree: CellLineageTree,
        ref_tree: CellLineageTree,
        file_name: str = "",
        width: int=300,
        show_leaf_name: bool = True):
    from ete3 import CircleFace, TreeStyle, NodeStyle, RectFace
    print(file_name)
    tree.ladderize()
    ref_tree.ladderize()

    nstyle = NodeStyle()
    nstyle["size"] = 0
    for n in tree.traverse():
        if not n.is_leaf():
            n.set_style(nstyle)

    leaf_dict = {}
    for leaf_idx, leaf in enumerate(ref_tree):
        leaf_dict[leaf.allele_events_list_str] = leaf_idx
    for leaf_idx, leaf in enumerate(tree):
        leaf.name = "%d" % leaf_dict[leaf.allele_events_list_str]

    tree.show_leaf_name = show_leaf_name

    tree.show_branch_length = True
    ts = TreeStyle()
    ts.scale = 100

    tree.render(file_name, w=width, units="mm", tree_style=ts)
    print("done")

def gather_results(
        get_true_model_fnc,
        get_result_fnc,
        get_rand_tree_fnc,
        seeds,
        settings,
        n_bcode = 1,
        tree_idx = 1,
        num_rands = 10,
        do_plots = False,
        print_keys = [],
        out_true_mrca_plot = None,
        out_fitted_mrca_plot = None):
    get_param_func_dict = {
            "bhv": None,
            "random_bhv": None,
            "zero_bhv": None,
            "super_zero_bhv": None,
            "internal_corr": None,
            "internal_random_corr": None,
            "leaves": None,
            "seeds": None,
            "only_targ": get_only_target_lams,
            "targ": get_target_lams,
            "double": get_double_cut}

    n_bcode_results = {
            key: [[] for _ in settings]
            for key in get_param_func_dict.keys()}

    for key in get_param_func_dict.keys():
        get_param_func = get_param_func_dict[key]
        if get_param_func is None:
            continue

        for seed in seeds:
            for idx, setting in enumerate(settings):
                try:
                    true_model = get_true_model_fnc(seed, setting, n_bcode)
                except FileNotFoundError as e:
                    print(e)
                    continue
                true_model_val = get_param_func(true_model)
                try:
                    result = get_result_fnc(seed, setting, n_bcode)
                except FileNotFoundError as e:
                    print(e)
                    continue
                fitted_val = get_param_func(result)
                dist = np.linalg.norm(fitted_val - true_model_val, ord=1)/np.linalg.norm(true_model_val, ord=1)
                n_bcode_results[key][idx].append(dist)

    for seed_idx, seed in enumerate(seeds):
        for idx, setting in enumerate(settings):
            try:
                true_model = get_true_model_fnc(seed, setting, n_bcode)
                tot_height = true_model[0]["tot_time"]
            except FileNotFoundError:
                continue
            true_internal_meas = InternalCorrMeasurer(true_model[tree_idx], "_output/scratch")
            true_mrca_meas = MRCADistanceMeasurer(true_model[tree_idx], "_output/scratch")
            true_bhv_meas = BHVDistanceMeasurer(true_model[tree_idx], "_output/scratch")

            if seed_idx == 0 and do_plots:
                #plot_mrca_matrix(
                #    true_mrca_meas.ref_tree_mrca_matrix,
                #    out_true_mrca_plot)
                plot_tree(
                    true_model[tree_idx],
                    true_model[tree_idx],
                    out_true_mrca_plot)

            try:
                result = get_result_fnc(seed, setting, n_bcode)
            except FileNotFoundError:
                continue
            n_bcode_results["seeds"][idx].append(seed)
            n_bcode_results["leaves"][idx].append(len(result[2]))
            if seed_idx == 0 and do_plots:
                #plot_mrca_matrix(
                #    true_mrca_meas._get_mrca_matrix(result[tree_idx]),
                #    out_fitted_mrca_plot % setting)
                plot_tree(
                    result[tree_idx],
                    true_model[tree_idx],
                    out_fitted_mrca_plot % setting)

            try:
                rand_tree = get_rand_tree_fnc(seed, setting, n_bcode)
            except FileNotFoundError:
                continue

            rand_dists = []
            rand_bhv_dists = []
            for _ in range(num_rands):
                assign_rand_tree_lengths(rand_tree[tree_idx], tot_height)

                dist = true_internal_meas.get_dist(rand_tree[tree_idx])
                rand_dists.append(dist)
                dist = true_bhv_meas.get_dist(rand_tree[tree_idx])
                rand_bhv_dists.append(dist)
            n_bcode_results["random_bhv"][idx].append(np.mean(rand_bhv_dists))
            n_bcode_results["internal_random_corr"][idx].append(np.mean(rand_dists))

            zero_tree = rand_tree[tree_idx].copy()
            for node in zero_tree.traverse():
                if node.is_root():
                    continue
                if node.is_leaf():
                    node.dist = 1 - node.up.get_distance(zero_tree)
                    assert node.dist > 0
                else:
                    node.dist = 1e-10
            dist = true_bhv_meas.get_dist(zero_tree)
            n_bcode_results["zero_bhv"][idx].append(dist)

            for node in zero_tree:
                node.dist = 0
            dist = true_bhv_meas.get_dist(zero_tree)
            n_bcode_results["super_zero_bhv"][idx].append(dist)

            n_bcode_results["seeds"][idx].append(len(true_model[tree_idx]))

            dist = true_internal_meas.get_dist(result[tree_idx])
            n_bcode_results["internal_corr"][idx].append(dist)

            dist = true_bhv_meas.get_dist(result[tree_idx])
            n_bcode_results["bhv"][idx].append(dist)

    print_results(
            settings,
            n_bcode_results,
            n_bcode,
            print_keys)
