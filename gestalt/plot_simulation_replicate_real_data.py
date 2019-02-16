import numpy as np
import sys
import argparse
import os.path
import six
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from common import parse_comma_str
from anc_state import AncState


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot descriptive stats of real data vs. simulated data')
    parser.add_argument(
        '--real-data',
        type=str,
        default="analyze_gestalt/_output/dome1_abund1/fish_data_restrict.pkl")
    parser.add_argument(
        '--obs-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/num_barcodes1/obs_data.pkl")
    parser.add_argument(
        '--simulation-folder',
        type=str,
        default="simulation_replicate_real_data")
    parser.add_argument(
        '--model-seed',
        type=int,
        default=0)
    parser.add_argument(
        '--data-seeds',
        type=str,
        default="0")
    parser.add_argument(
        '--growth-stage',
        type=str,
        default="dome")
    parser.add_argument(
        '--out-plot-abundance',
        type=str,
        default="_output/simulation_abundance.png")
    parser.add_argument(
        '--out-plot-target-deact',
        type=str,
        default="_output/simulation_target_deact.png")
    parser.add_argument(
        '--out-plot-bcode-exhaust',
        type=str,
        default="_output/simulation_bcode_exhaust.png")

    parser.set_defaults()
    args = parser.parse_args(args)
    args.data_seeds = parse_comma_str(args.data_seeds, int)
    return args

def get_abundance(obs_data_dict):
    return [obs.abundance for obs in obs_data_dict["obs_leaves"]]

def plot_abundance_histogram(sim_obs_data_dict, real_obs_data_dict, out_plot_file):
    sim_abund = get_abundance(sim_obs_data_dict)
    real_abund = get_abundance(real_obs_data_dict)
    plt.clf()
    plt.hist([sim_abund, real_abund], bins=20, density=True, label=["Simulated", "Dome Fish 1"])
    plt.yscale('log')
    plt.xlabel("Number of times allele is observed")
    plt.ylabel("Frequency")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot_file)

def get_target_deactivated(obs_data_dict):
    bcode_meta = obs_data_dict["bcode_meta"]
    targets_deactivated = []
    for i in range(bcode_meta.num_barcodes):
        anc_states = [AncState.create_for_observed_allele(obs.allele_events_list[i], bcode_meta) for obs in obs_data_dict["obs_leaves"]]
        targ_statuses = [anc_state.to_max_target_status() for anc_state in anc_states]
        for targ_stat in targ_statuses:
            targets_deactivated += targ_stat.deact_targets
    return targets_deactivated

def plot_target_deact_histogram(sim_obs_data_dict, real_obs_data_dict, out_plot_file):
    sim_targ = get_target_deactivated(sim_obs_data_dict)
    real_targ = get_target_deactivated(real_obs_data_dict)
    plt.clf()
    plt.hist([sim_targ, real_targ], bins=10, density=True, label=["Simulated", "Dome Fish 1"])
    plt.xlabel("Target index")
    plt.ylabel("Frequency target was inactive")
    #plt.legend().remove()
    plt.tight_layout()
    plt.savefig(out_plot_file)

def get_bcode_exhaustion(obs_data_dict):
    bcode_meta = obs_data_dict["bcode_meta"]
    n_target_used_count = np.zeros(bcode_meta.n_targets + 1)
    for i in range(bcode_meta.num_barcodes):
        anc_states = [AncState.create_for_observed_allele(obs.allele_events_list[i], bcode_meta) for obs in obs_data_dict["obs_leaves"]]
        targ_statuses = [anc_state.to_max_target_status() for anc_state in anc_states]
        num_deacts = [targ_stat.num_deact_targets for targ_stat in targ_statuses]
        for n_deacts in num_deacts:
            n_target_used_count[n_deacts] += 1
    deact_proportions = n_target_used_count/np.sum(n_target_used_count)
    return deact_proportions

def plot_bcode_exhaustion(sim_obs_data_dict, real_obs_data_dict, out_plot_file):
    bcode_meta = sim_obs_data_dict["bcode_meta"]
    sim_deact_proportions = get_bcode_exhaustion(sim_obs_data_dict)
    real_deact_proportions = get_bcode_exhaustion(real_obs_data_dict)

    data = pd.DataFrame.from_dict({
        "num_targets_deactivated": np.concatenate([np.arange(bcode_meta.n_targets + 1), np.arange(bcode_meta.n_targets + 1)]),
        "proportions": np.concatenate([sim_deact_proportions, real_deact_proportions]),
        "Data": ["Simulated"] * (bcode_meta.n_targets + 1) + ["Dome Fish 1"] * (bcode_meta.n_targets + 1)
    })
    plt.clf()
    sns.catplot(
        x="num_targets_deactivated",
        y="proportions",
        hue="Data",
        data=data,
        kind="bar")
    plt.yscale('log')
    plt.xlabel("Number of inactive targets in observed barcode")
    plt.ylabel("Proportion")
    plt.savefig(out_plot_file)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    with open(args.real_data, "rb") as f:
        real_obs_data_dict = six.moves.cPickle.load(f)

    sns.set_context("paper", font_scale = 1.4)
    for data_seed in args.data_seeds:
        obs_file = os.path.join(
                args.simulation_folder,
                args.obs_file_template % (args.model_seed, data_seed, args.growth_stage))
        with open(obs_file, "rb") as f:
            sim_obs_data_dict = six.moves.cPickle.load(f)

        plot_abundance_histogram(sim_obs_data_dict, real_obs_data_dict, args.out_plot_abundance)
        plot_target_deact_histogram(sim_obs_data_dict, real_obs_data_dict, args.out_plot_target_deact)
        plot_bcode_exhaustion(sim_obs_data_dict, real_obs_data_dict, args.out_plot_bcode_exhaust)


if __name__ == "__main__":
    main()
