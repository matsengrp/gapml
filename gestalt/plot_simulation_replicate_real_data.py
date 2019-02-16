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

# number of targets in real barcode
NUM_TARGETS = 10
# line width for the real data
REAL_SIZE = 4

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
        default=",".join(map(str, range(1,11))))
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

def get_abundance_freqs(obs_data_dict, label, style, max_abundance=30):
    abundances = np.reshape(np.array([obs.abundance for obs in obs_data_dict["obs_leaves"]]), (-1,1))
    num_uniq_alleles = abundances.size
    thresholds = np.reshape(np.arange(max_abundance), (1,-1))
    abundance_freqs = np.sum(abundances <= thresholds, axis=0)
    return pd.DataFrame({
            "CDF": (abundance_freqs/num_uniq_alleles).flatten(),
            "num_obs": np.arange(max_abundance),
            "label": label,
            "style": style})

def plot_abundance_histogram(sim_obs_all, real_obs_data_dict, out_plot_file):
    all_data = []
    for idx, sim_obs_data_dict in enumerate(sim_obs_all):
        sim_abund = get_abundance_freqs(
                sim_obs_data_dict,
                "Simulation %d" % idx,
                "Simulation")
        all_data.append(sim_abund)
    real_abund = get_abundance_freqs(real_obs_data_dict,
            "Dome Fish 1",
            "Real")
    all_data.append(real_abund)
    df = pd.concat(all_data)

    plt.clf()
    sns.lineplot(
            x="num_obs",
            y="CDF",
            hue="label",
            data=df,
            size="style",
            sizes={
                "Simulation": 1,
                "Real": REAL_SIZE},
            legend=False)
    plt.xlabel("Number of times allele is observed")
    plt.tight_layout()
    plt.savefig(out_plot_file)

def get_target_deactivated_df(obs_data_dict, label, style):
    bcode_meta = obs_data_dict["bcode_meta"]
    targets_deactivated = []
    for i in range(bcode_meta.num_barcodes):
        anc_states = [AncState.create_for_observed_allele(obs.allele_events_list[i], bcode_meta) for obs in obs_data_dict["obs_leaves"]]
        targ_statuses = [anc_state.to_max_target_status() for anc_state in anc_states]
        for targ_stat in targ_statuses:
            targets_deactivated += targ_stat.deact_targets
    targets_deactivated = np.reshape(np.array(targets_deactivated), (-1,1))
    denom = targets_deactivated.size
    num_deactivated = np.reshape(np.arange(bcode_meta.n_targets), (1, -1))
    deact_freqs = np.sum(targets_deactivated == num_deactivated, axis=0)/denom
    return pd.DataFrame({
        "Frequency": deact_freqs,
        "target": np.arange(bcode_meta.n_targets) + 1,
        "label": label,
        "style": style})

def plot_target_deact_histogram(sim_obs_all, real_obs_data_dict, out_plot_file):
    all_data = []
    for idx, sim_obs_data_dict in enumerate(sim_obs_all):
        sim_targ = get_target_deactivated_df(
                sim_obs_data_dict,
                "Simulated %d" % idx,
                "Simulation")
        all_data.append(sim_targ)
    real_targ = get_target_deactivated_df(
            real_obs_data_dict,
            "Dome Fish 1",
            "Real")
    all_data.append(real_targ)
    df = pd.concat(all_data)

    plt.clf()
    #plt.hist([sim_targ, real_targ], bins=10, density=True, label=["Simulated", "Dome Fish 1"])
    #sns.kdeplot(sim_targ, label="Simulated")
    #sns.kdeplot(real_targ, label="Dome Fish 1")
    sns.lineplot(
            x="target",
            y="Frequency",
            hue="label",
            data=df,
            size="style",
            sizes={
                "Simulation": 1,
                "Real": REAL_SIZE},
            legend=False)
    plt.xlabel("Target index")
    plt.xticks(np.arange(1, NUM_TARGETS + 1))
    plt.ylabel("Frequency target was deactivated")
    plt.tight_layout()
    plt.savefig(out_plot_file)

def get_bcode_exhaustion_df(obs_data_dict, label, style):
    bcode_meta = obs_data_dict["bcode_meta"]
    n_target_used_count = np.zeros(bcode_meta.n_targets + 1)
    for i in range(bcode_meta.num_barcodes):
        anc_states = [AncState.create_for_observed_allele(obs.allele_events_list[i], bcode_meta) for obs in obs_data_dict["obs_leaves"]]
        targ_statuses = [anc_state.to_max_target_status() for anc_state in anc_states]
        num_deacts = [targ_stat.num_deact_targets for targ_stat in targ_statuses]
        for n_deacts in num_deacts:
            n_target_used_count[n_deacts] += 1
    deact_proportions = n_target_used_count/np.sum(n_target_used_count)
    return pd.DataFrame({
        "num_targets_deactivated": np.arange(bcode_meta.n_targets + 1),
        "proportions": deact_proportions,
        "label": label,
        "style": style})

def plot_bcode_exhaustion(sim_obs_all, real_obs_data_dict, out_plot_file):
    all_data = []
    for idx, sim_obs_data_dict in enumerate(sim_obs_all):
        sim_deact_proportions = get_bcode_exhaustion_df(
                sim_obs_data_dict,
                "Simulated %d" % idx,
                "Simulation")
        all_data.append(sim_deact_proportions)
    real_deact_proportions = get_bcode_exhaustion_df(
            real_obs_data_dict,
            "Dome Fish 1",
            "Real")
    all_data.append(real_deact_proportions)
    data = pd.concat(all_data)

    plt.clf()
    sns.lineplot(
        x="num_targets_deactivated",
        y="proportions",
        hue="label",
        data=data,
        size="style",
        sizes={
            "Simulation": 1,
            "Real": REAL_SIZE},
        legend=False)
    #plt.yscale('log')
    plt.xlabel("Number of inactive targets in observed barcode")
    plt.xticks(np.arange(NUM_TARGETS + 1))
    plt.ylabel("Proportion")
    plt.savefig(out_plot_file)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    with open(args.real_data, "rb") as f:
        real_obs_data_dict = six.moves.cPickle.load(f)
        print("num obs real", len(real_obs_data_dict["obs_leaves"]))
        num_obs_real = len(real_obs_data_dict["obs_leaves"])

    sim_obs_all = []
    for data_seed in args.data_seeds:
        obs_file = os.path.join(
                args.simulation_folder,
                args.obs_file_template % (args.model_seed, data_seed, args.growth_stage))
        with open(obs_file, "rb") as f:
            sim_obs_data_dict = six.moves.cPickle.load(f)
            print("num obs SIM", len(sim_obs_data_dict["obs_leaves"]))
        if len(sim_obs_data_dict["obs_leaves"]) > num_obs_real * 0.9:
            sim_obs_all.append(sim_obs_data_dict)
        else:
            print("skipping. too few obs")

    sns.set_context("paper", font_scale=1.4)
    plot_abundance_histogram(sim_obs_all, real_obs_data_dict, args.out_plot_abundance)
    plot_target_deact_histogram(sim_obs_all, real_obs_data_dict, args.out_plot_target_deact)
    plot_bcode_exhaustion(sim_obs_all, real_obs_data_dict, args.out_plot_bcode_exhaust)


if __name__ == "__main__":
    main()
