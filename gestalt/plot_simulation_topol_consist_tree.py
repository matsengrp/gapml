import sys
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from cell_lineage_tree import CellLineageTree
from common import parse_comma_str
from tree_distance import TreeDistanceMeasurerAgg, InternalCorrMeasurer, BHVDistanceMeasurer
from plot_simulation_topol_consist import get_true_model, get_mle_result, get_chronos_result, get_neighbor_joining_result


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot individual trees and internal node heights')
    parser.add_argument(
        '--true-model-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/true_model.pkl")
    parser.add_argument(
        '--mle-file-template',
        type=str,
        #default="_output/model_seed%d/%d/%s/num_barcodes%d/sum_states_10/extra_steps_1/tune_fitted_pretuned.pkl")
        default="_output/model_seed%d/%d/%s/num_barcodes%d/sum_states_10/extra_steps_1/tune_fitted.pkl")
    parser.add_argument(
        '--chronos-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/num_barcodes%d/chronos_fitted.pkl")
    parser.add_argument(
        '--nj-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/num_barcodes%d/nj_fitted.pkl")
    parser.add_argument(
        '--model-seed',
        type=int,
        default=100)
    parser.add_argument(
        '--data-seed',
        type=int,
        default=400)
    parser.add_argument(
        '--n-bcodes-list',
        type=str,
        default="1,2,4")
    parser.add_argument(
        '--simulation-folder',
        type=str,
        default="simulation_compare")
    parser.add_argument(
        '--growth-stage',
        type=str,
        default="small")
    parser.add_argument(
        '--out-plot-template',
        type=str,
        default="_output/simulation_tree_%d_%d_%s.png")
    parser.add_argument(
        '--out-heights-plot-template',
        type=str,
        default="_output/simulation_tree_heights_%d.png")
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")

    parser.set_defaults()
    args = parser.parse_args(args)

    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.n_bcodes_list = parse_comma_str(args.n_bcodes_list, int)
    return args

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
        leaf_dict[leaf.leaf_key] = leaf_idx
    for leaf_idx, leaf in enumerate(tree):
        print(leaf.leaf_key)
    for leaf_idx, leaf in enumerate(tree):
        leaf.name = "%d" % leaf_dict[leaf.leaf_key]
    print(len(tree), len(ref_tree))
    assert len(tree) == len(ref_tree)

    tree.show_leaf_name = show_leaf_name

    tree.show_branch_length = True
    ts = TreeStyle()
    ts.scale = 100

    tree.render(file_name, w=width, units="mm", tree_style=ts)

def plot_internal_node_heights(internal_node_heights, file_name):
    print(file_name)
    sns_plot = sns.lmplot(
            x="true",
            y="fitted",
            hue="method",
            col="n_bcodes",
            row="assess",
            data=internal_node_heights,
            aspect=1,
            x_jitter=0.01,
            y_jitter=0.02,
            fit_reg=False,
            markers=["o", "x", "^"])
    sns_plot.savefig(file_name)

def main(args=sys.argv[1:]):
    args = parse_args(args)

    internal_node_heights = []
    for n_bcodes in args.n_bcodes_list:
        print("barcodes", n_bcodes)
        true_params, assessor = get_true_model(
                args,
                args.data_seed,
                n_bcodes,
                measurer_classes=[InternalCorrMeasurer, BHVDistanceMeasurer])
        mle_params, mle_tree = get_mle_result(args, args.data_seed, n_bcodes)
        full_mle_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(mle_tree, "leaf_key")
        full_tree_assessor = assessor._get_full_tree_assessor(mle_tree)
        collapse_tree_assessor, collapse_mle_tree_raw = assessor._get_collapse_tree_assessor(mle_tree)
        collapse_mle_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(collapse_mle_tree_raw, "leaf_key")
        _, chronos_tree = get_chronos_result(args, args.data_seed, n_bcodes, assessor)
        full_chronos_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(chronos_tree, "leaf_key")
        _, collapse_chronos_tree_raw = assessor._get_collapse_tree_assessor(chronos_tree)
        collapse_chronos_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(collapse_chronos_tree_raw, "leaf_key")
        _, neighbor_joining_tree = get_neighbor_joining_result(args, args.data_seed, n_bcodes, assessor)
        full_neighbor_joining_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(neighbor_joining_tree, "leaf_key")
        _, collapse_neighbor_joining_tree_raw = assessor._get_collapse_tree_assessor(neighbor_joining_tree)
        collapse_neighbor_joining_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(collapse_neighbor_joining_tree_raw, "leaf_key")

        #if n_bcodes == args.n_bcodes_list[0]:
        #    plot_tree(
        #        full_tree_assessor.ref_tree,
        #        full_tree_assessor.ref_tree,
        #        args.out_plot_template % (0, args.data_seed, "true"))

        #plot_tree(
        #    full_mle_tree,
        #    full_tree_assessor.ref_tree,
        #    args.out_plot_template % (n_bcodes, args.data_seed, "mle"))

        #plot_tree(
        #    full_chronos_tree,
        #    full_tree_assessor.ref_tree,
        #    args.out_plot_template % (n_bcodes, args.data_seed, "chronos"))

        #plot_tree(
        #    full_neighbor_joining_tree,
        #    full_tree_assessor.ref_tree,
        #    args.out_plot_template % (n_bcodes, args.data_seed, "neighbor_joining"))

        tree_assessor_dict = {
                "full": (full_tree_assessor, full_mle_tree, full_chronos_tree, full_neighbor_joining_tree),
                "collapse": (collapse_tree_assessor, collapse_mle_tree, collapse_chronos_tree, collapse_neighbor_joining_tree)}
        for assessor_name, (tree_assessor, m_tree, c_tree, nj_tree) in tree_assessor_dict.items():
            print(assessor_name)
            intern_measurer = tree_assessor.measurers[0]
            m_tree.label_dist_to_roots()
            mle_heights = intern_measurer.get_compare_node_distances(
                m_tree,
                intern_measurer.ref_leaf_groups)
            internal_node_heights.append(pd.DataFrame.from_dict({
                "method": "mle",
                "assess": assessor_name,
                "n_bcodes": n_bcodes,
                "true": intern_measurer.ref_node_val,
                "fitted": mle_heights}))
            c_tree.label_dist_to_roots()
            chronos_heights = intern_measurer.get_compare_node_distances(
                c_tree,
                intern_measurer.ref_leaf_groups)
            internal_node_heights.append(pd.DataFrame.from_dict({
                "method": "chronos",
                "assess": assessor_name,
                "n_bcodes": n_bcodes,
                "true": intern_measurer.ref_node_val,
                "fitted": chronos_heights}))
            nj_tree.label_dist_to_roots()
            chronos_heights = intern_measurer.get_compare_node_distances(
                nj_tree,
                intern_measurer.ref_leaf_groups)
            internal_node_heights.append(pd.DataFrame.from_dict({
                "method": "nj",
                "assess": assessor_name,
                "n_bcodes": n_bcodes,
                "true": intern_measurer.ref_node_val,
                "fitted": chronos_heights}))
    internal_node_heights = pd.concat(internal_node_heights)
    plot_internal_node_heights(
            internal_node_heights,
            args.out_heights_plot_template % args.data_seed)


if __name__ == "__main__":
    main()
