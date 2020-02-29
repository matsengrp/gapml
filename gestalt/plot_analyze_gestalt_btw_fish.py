"""
Plot the target cut rates for the different fish for comparison.
Also runs hypothesis tests using bootstrap
"""

import six
import sys
import argparse
import os.path
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from common import parse_comma_str

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='compare and plot mutation params for different fish')
    parser.add_argument(
        '--obs-file-template',
        type=str,
        default="_output/%s/sampling_seed0/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-templates',
        type=str,
        default="_output/%s/sampling_seed0/sum_states_20/extra_steps_1/tune_pen_hanging.pkl")
    parser.add_argument(
        '--num-rands',
        type=int,
        default=2000)
    parser.add_argument(
        '--fish-names',
        type=str,
        #default="30hpf")
        default="4.3hpf")
        #default="3day")
        #default="30hpf+3day")
    parser.add_argument(
        '--fishies',
        type=str,
        #default="ADR1,ADR2")
        #default="30hpf_v6_4,30hpf_v6_5,30hpf_v6_6,30hpf_v6_8")
        #default="epi90_2,epi90_5,epi90_8,epi90_9,epi90_10,epi90_12")
        default="dome1,dome3,dome8,dome10")
        #default="3day1,3day2,3day3,3day4,3day5,3day6")
    parser.add_argument(
        '--out-plot-template',
        type=str,
        default="_output/target_lam_compare_%s.png")
    parser.set_defaults()
    args = parser.parse_args(args)
    if "+" in args.fishies:
        fishies1, fishies2 = args.fishies.split("+")
        args.fishies = [parse_comma_str(fishies1, str), parse_comma_str(fishies2, str)]
        args.fish_names = args.fish_names.split("+")
    else:
        args.fishies = [parse_comma_str(args.fishies, str)]
        args.fish_names = [args.fish_names]
    args.mle_templates = parse_comma_str(args.mle_templates, str)
    return args

def _estimate_targ_rate_simple(obs_data_dict):
    """
    Just count up the number of times the outermost target was cut
    in unique observed indels
    """
    bcode_meta = obs_data_dict["bcode_meta"]
    obs_data = obs_data_dict["obs_leaves"]
    all_events = [o.allele_events_list[0].events for o in obs_data]
    uniq_events = set([evt for evts in all_events for evt in evts])
    targ_used = np.zeros(bcode_meta.n_targets)
    for evt in uniq_events:
        if evt.min_target != evt.max_target:
            targ_used[evt.min_target] += 1
            targ_used[evt.max_target] += 1
        else:
            targ_used[evt.min_target] += 2
    return targ_used/np.sum(targ_used)

def _get_all_pairwise_correlations(param_vals):
    """
    Return pairwise Spearman correlations
    """
    num_obs = len(param_vals)
    all_corrs = np.zeros((num_obs, num_obs))
    for idx1, fit_param1 in enumerate(param_vals):
        for idx2_offset, fit_param2 in enumerate(param_vals[idx1:]):
            corr = scipy.stats.spearmanr(fit_param1, fit_param2)[0]
            idx2 = idx1 + idx2_offset
            all_corrs[idx1,idx2] = corr
            all_corrs[idx2,idx1] = corr
    print("all_corr", all_corrs)
    return all_corrs

def _get_two_sample_pairwise_correlations(param_vals1, param_vals2):
    """
    Return pairwise Spearman correlations
    """
    num_obs1 = len(param_vals1)
    num_obs2 = len(param_vals2)
    all_corrs = np.zeros((num_obs1, num_obs2))
    for idx1, fit_param1 in enumerate(param_vals1):
        for idx2, fit_param2 in enumerate(param_vals2):
            corr = scipy.stats.spearmanr(fit_param1, fit_param2)[0]
            all_corrs[idx1,idx2] = corr
    print("all_corr", all_corrs)
    return all_corrs

def _get_mean_corr(corr_matrix):
    """
    Take the upper triangular matrix above the diagonal
    """
    num_obs = corr_matrix.shape[0]
    return np.mean(corr_matrix[np.triu_indices(num_obs, k=1)])

def _get_summary_bootstrap(all_corrs, num_rands=5, ci_limits=[2.5, 97.5]):
    """
    We bootstrap fish to get bootstrap confidence intervals
    (Bootstrap is totally valid because we are trying to calculate a U-statistic!)
    """
    mean_corr = _get_mean_corr(all_corrs)

    num_obs = all_corrs.shape[0]
    if num_obs > 2:
        bootstrap_vals = []
        for _ in range(num_rands):
            sampled_obs = np.random.choice(num_obs, num_obs, replace=True)
            new_corr_matrix = np.zeros((num_obs, num_obs))
            for idx1, i in enumerate(sampled_obs):
                for idx2_offset, j in enumerate(sampled_obs[idx1 + 1:]):
                    idx2 = idx1 + 1 + idx2_offset
                    new_corr_matrix[idx1, idx2] = all_corrs[i, j]
            bootstrap_mean_corr = _get_mean_corr(new_corr_matrix)
            bootstrap_vals.append(bootstrap_mean_corr)
        bootstrap_ci = np.percentile(bootstrap_vals, ci_limits)
    else:
        bootstrap_ci = np.array([mean_corr, mean_corr])
    quantiles = np.percentile(all_corrs[np.triu_indices(num_obs, k=1)], ci_limits)
    return mean_corr, bootstrap_ci, quantiles

def _get_summary_two_sample_bootstrap(all_corrs, num_rands=5, ci_limits=[2.5, 97.5]):
    """
    We bootstrap fish to get bootstrap confidence intervals
    (Bootstrap is totally valid because we are trying to calculate a U-statistic!)
    """
    mean_corr = _get_mean_corr(all_corrs)

    num_obs = all_corrs.shape
    if num_obs[0] >= 2 and num_obs[1] >= 2:
        bootstrap_vals = []
        for _ in range(num_rands):
            sampled_obs1 = np.random.choice(num_obs[0], num_obs[0], replace=True)
            sampled_obs2 = np.random.choice(num_obs[1], num_obs[1], replace=True)
            new_corr_matrix = np.zeros(num_obs)
            for idx1, i in enumerate(sampled_obs1):
                for idx2, j in enumerate(sampled_obs2):
                    new_corr_matrix[idx1, idx2] = all_corrs[i, j]
            bootstrap_mean_corr = _get_mean_corr(new_corr_matrix)
            bootstrap_vals.append(bootstrap_mean_corr)
        bootstrap_ci = np.percentile(bootstrap_vals, ci_limits)
    else:
        bootstrap_ci = np.array([mean_corr, mean_corr])
    quantiles = np.percentile(all_corrs, ci_limits)
    return mean_corr, bootstrap_ci, quantiles

def load_my_fish(fish, args):
    print(fish)
    # Load our estimated target rates
    for mle_file_template in args.mle_templates:
        file_name = mle_file_template % fish
        if os.path.exists(file_name):
            break
    with open(file_name, "rb") as f:
        fitted_data = six.moves.cPickle.load(f)
        if "final_fit" in fitted_data:
            res = fitted_data["final_fit"]
        else:
            res = fitted_data[-1]["best_res"]
        fitted_param = res.model_params_dict["target_lams"]

    # Fit a simple target rate thing
    file_name = args.obs_file_template % fish
    with open(file_name, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)
    return fitted_param, obs_data_dict

def _get_target_lam_cis(param_vals):
    all_cis = []
    for i in range(len(param_vals[0])):
        target_lam_estimates = [np.log(estimates[i]) for estimates in param_vals]
        mean_target_lam_est = np.mean(target_lam_estimates)
        se_target_lam_est = np.sqrt(np.var(target_lam_estimates)/len(param_vals))
        all_cis.append([np.exp(mean_target_lam_est - 1.96 * se_target_lam_est), np.exp(mean_target_lam_est + 1.96 * se_target_lam_est)])
    return all_cis

def main(args=sys.argv[1:]):
    args = parse_args(args)
    all_fitted_params = []
    for fish_category, fish_name in zip(args.fishies, args.fish_names):
        fitted_params = {
            "mle": [],
            "simple": []}
        all_fitted_params.append(fitted_params)
        for fish in fish_category:
            try:
                fitted_param, obs_data_dict = load_my_fish(fish, args)
            except FileNotFoundError as e:
                print("not found %s" % str(e))
                continue

            fitted_params["mle"].append(fitted_param)
            fitted_params["simple"].append(_estimate_targ_rate_simple(obs_data_dict))

        for key in ["mle", "simple"]:
            print("fitting method", key, "fish", fish_name)
            param_vals = fitted_params[key]
            all_cis = _get_target_lam_cis(param_vals)
            print(all_cis)
            all_corrs = _get_all_pairwise_correlations(param_vals)
            corr_mean, ci_95, quantiles_95 = _get_summary_bootstrap(all_corrs, args.num_rands)
            print("Mean %.03f" % corr_mean)
            print("95 CI (%.03f, %.03f)" % (ci_95[0], ci_95[1]))
            print("95 quantiles (%.03f, %.03f)" % (quantiles_95[0], quantiles_95[1]))

        # Plot all the params
        pd_data = []
        for idx, param in enumerate(fitted_params["mle"]):
            for targ_idx, targ_val in enumerate(param):
                pd_data.append({
                    "target": targ_idx,
                    "val": targ_val,
                    "fish": idx + 1})
        data = pd.DataFrame(pd_data)
        sns.set_context("paper", font_scale=1.8)
        sns.lineplot(x="target", y="val", hue="fish", data=data, legend=False)
        plt.xlabel("Target index")
        plt.ylabel("Estimated cut rate")
        plt.yscale("log")
        plt.title(fish_name)
        plt.tight_layout()
        sns.despine()
        plt.savefig(args.out_plot_template % fish_name.replace(" ","_"))

    if len(args.fishies) == 2:
        print("TWO SAMPLE STUFF NOW")
        all_two_sample_corrs = _get_two_sample_pairwise_correlations(all_fitted_params[0]["mle"], all_fitted_params[1]["mle"])
        corr_mean, ci_95, quantiles_95 = _get_summary_two_sample_bootstrap(all_two_sample_corrs, args.num_rands)
        print("Mean %.03f" % corr_mean)
        print("95 CI (%.03f, %.03f)" % (ci_95[0], ci_95[1]))
        print("95 quantiles (%.03f, %.03f)" % (quantiles_95[0], quantiles_95[1]))

if __name__ == "__main__":
    main()
