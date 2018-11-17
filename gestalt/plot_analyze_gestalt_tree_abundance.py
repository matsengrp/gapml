"""
Plot trees for young embryos. not adult. no cell types.
"""
import os
import sys
import argparse
import six
import numpy as np

import matplotlib
matplotlib.use('Agg')

from plot_mrca_matrices import plot_tree
import collapsed_tree

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot tree')
    parser.add_argument(
        '--obs-file-template',
        type=str,
        default="_output/%s/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-file-template',
        type=str,
        default="_output/%s/sum_states_10/extra_steps_0/tune_pen.pkl")
    parser.add_argument(
        '--folder',
        type=str,
        default="analyze_gestalt")
    parser.add_argument(
        '--fish',
        type=str,
        default="dome5_abund1")
    parser.add_argument(
        '--out-plot-template',
        type=str,
        default="_output/tree_abund_%s_noname.png")
    args = parser.parse_args(args)
    return args

def load_data(args):
    fitted_tree_file = os.path.join(args.folder, args.mle_file_template % args.fish)
    with open(fitted_tree_file, "rb") as f:
        fitted_bifurc_tree = six.moves.cPickle.load(f)["final_fit"].fitted_bifurc_tree
    return fitted_bifurc_tree

def plot_gestalt_tree(
        fitted_bifurc_tree,
        out_plot_file,
        coll_dist=0.001):
    from ete3 import NodeStyle, RectFace
    print("at plotting phase....")
    #for l in fitted_bifurc_tree:
    for leaf in fitted_bifurc_tree:
        nstyle = NodeStyle()
        nstyle["size"] = 0
        leaf.set_style(nstyle)

        leaf.name = "" # leaf.allele_events_list_str

        seqFace = RectFace(
            width=np.log2(leaf.abundance) + 1,
            height=0.2,
            fgcolor="blue",
            bgcolor="blue")
        leaf.add_face(seqFace, 0, position="aligned")

    # Collapse distances for plot readability
    for node in fitted_bifurc_tree.get_descendants():
        if node.dist < coll_dist:
            node.dist = 0
    col_tree = collapsed_tree.collapse_zero_lens(fitted_bifurc_tree)

    plot_tree(
            col_tree,
            out_plot_file,
            width=1600,
            height=2500,
            show_leaf_name=False)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    # TODO: this doesnt work right now. need to add in prefix of tmp_mount
    tree = load_data(args)
    out_plot = args.out_plot_template % args.fish
    plot_gestalt_tree(
        tree,
        out_plot)


if __name__ == "__main__":
    main()
