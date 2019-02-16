"""
Plot the results of simulation_many_vs_one_bcode.
X-axis: num barcodes
Y-axis: num unique alleles
"""

import sys
import argparse
import random
import six
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import seaborn as sns

from common import parse_comma_str


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot results of simulation_many_vs_one_bcode')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--idxs',
        type=str,
        default="0",
        help='comma separated idx for obs files')
    parser.add_argument(
        '--obs-files-one',
        type=str,
        default="_output/obs_data_one.pkl",
        help="""
        comma separated pkl files containing the observed data -- this data was generated using
        a single bcode
        """)
    parser.add_argument(
        '--many-bcode-idxs',
        type=str,
        default="0",
        help='comma separated idx for obs files using many bcodes')
    parser.add_argument(
        '--obs-files-many',
        type=str,
        default="_output/obs_data_many.pkl",
        help="""
        comma separated pkl files containing the observed data -- this data was generated using
        many bcodes
        """)
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/parsimony_vs_likelihood.png",
        help='plot file name')
    parser.set_defaults()
    args = parser.parse_args(args)

    args.idxs = parse_comma_str(args.idxs, int)
    args.obs_files_one = parse_comma_str(args.obs_files_one, str)
    args.obs_files_many = parse_comma_str(args.obs_files_many, str)
    assert len(args.idxs) == len(args.obs_files_one)
    assert len(args.idxs) == len(args.obs_files_many)

    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_results = []
    # Read out the results
    for idx, obs_file_one, obs_file_many in zip(args.idxs, args.obs_files_one, args.obs_files_many):
        with open(obs_file_one, "rb") as f:
            obs_data_one = six.moves.cPickle.load(f)
            bcode_meta_one = obs_data_one["bcode_meta"]
            num_obs_one_bcode = len(obs_data_one["obs_leaves"])
            assert bcode_meta_one.num_barcodes == 1
        with open(obs_file_many, "rb") as f:
            obs_data_many = six.moves.cPickle.load(f)
            bcode_meta_many = obs_data_many["bcode_meta"]
            num_obs_many_bcodes = len(obs_data_many["obs_leaves"])
            assert bcode_meta_many.n_targets * bcode_meta_many.num_barcodes == bcode_meta_one.n_targets

        n_targets = bcode_meta_one.n_targets

        all_results.append({
            "idx": idx,
            "num_targets": n_targets,
            "num_obs_one": num_obs_one_bcode,
            "num_obs_many": num_obs_many_bcodes,
            "num_obs_delta": num_obs_many_bcodes - num_obs_one_bcode
        })

    all_results = pd.DataFrame(all_results)
    print(all_results)

    # Actually make the plot
    sns.set_context("paper", font_scale=1.4)
    sns.lineplot(
            x="num_targets",
            y="num_obs_delta",
            hue="idx",
            data=all_results,
            legend=False)
    pyplot.ylabel("Difference in number of unique alleles")
    pyplot.xlabel("Number of targets")
    pyplot.xticks(np.arange(6, 36, step=6))
    pyplot.tight_layout(pad=1.2)
    pyplot.savefig(args.out_plot)


if __name__ == "__main__":
    main()
