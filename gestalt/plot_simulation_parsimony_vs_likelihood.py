import sys
import argparse
import random
import six
import numpy as np
import pandas as pd

import ancestral_events_finder
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from common import parse_comma_str


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='make a less parsimonious tree')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--num-spr-moves',
        type=int,
        default=1)
    parser.add_argument(
        '--tree-idxs',
        type=str,
        default="0",
        help='comma separated idx for trees')
    parser.add_argument(
        '--tree-files',
        type=str,
        default="_output/tune_fitted.pkl",
        help='comma separated pkl files containing the fitted models')
    parser.add_argument(
        '--obs-files',
        type=str,
        default="_output/obs_data.pkl",
        help='comma separated pkl files containing the observed data -- only used to get barcode metadata')
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/parsimony_vs_likelihood.png",
        help='plot file name')
    parser.set_defaults()
    args = parser.parse_args(args)
    args.tree_idxs = parse_comma_str(args.tree_idxs, int)
    args.tree_files = parse_comma_str(args.tree_files, str)
    args.obs_files = parse_comma_str(args.obs_files, str)
    return args


def get_tree_parsimony_score(tree: CellLineageTree, bcode_meta: BarcodeMetadata):
    ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
    return ancestral_events_finder.get_parsimony_score(tree)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_results = []
    for tree_idx, tree_file, obs_file in zip(args.tree_idxs, args.tree_files, args.obs_files):
        with open(obs_file, "rb") as f:
            obs_data = six.moves.cPickle.load(f)
            bcode_meta = obs_data["bcode_meta"]
        with open(tree_file, "rb") as f:
            fitted_res = six.moves.cPickle.load(f)
        fitted_tree = fitted_res["final_fit"].fitted_bifurc_tree
        pen_log_lik = fitted_res["final_fit"].pen_log_lik[0]
        pars_score = get_tree_parsimony_score(fitted_tree, bcode_meta)
        all_results.append({
            "idx": tree_idx,
            "pen_log_lik": pen_log_lik,
            "parsimony_score": pars_score})

    all_results = pd.DataFrame(all_results)
    print(all_results)
    argmin_pars_score = all_results.groupby('idx')['parsimony_score'].idxmin()
    argmin_pars_score = argmin_pars_score.reset_index().rename(
            index=str,
            columns={"parsimony_score": "argmin_pars"})
    merged_df = all_results.merge(argmin_pars_score, on="idx")
    pll_min_pars = merged_df['pen_log_lik'].values[merged_df['argmin_pars']]
    merged_df['pll_delta'] = merged_df['pen_log_lik'] - pll_min_pars
    print(merged_df)


if __name__ == "__main__":
    main()
