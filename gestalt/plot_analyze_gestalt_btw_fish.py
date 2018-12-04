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
        '--mle-file-template',
        type=str,
        default="_output/%s/sampling_seed0/sum_states_25/extra_steps_1/tune_pen_hanging.pkl")
    parser.add_argument(
        '--folder',
        type=str,
        default="analyze_gestalt")
    parser.add_argument(
        '--num-rands',
        type=int,
        default=2000)
    parser.add_argument(
        '--fishies',
        type=str,
        #default="ADR1,ADR2")
        #default="30hpf_v6_5,30hpf_v6_6,30hpf_v6_7,30hpf_v6_8")
        default="dome1,dome3,dome8,dome10")
        #default="3day1,3day2,3day3,3day4,3day5")
    parser.add_argument(
        '--out-plot-file',
        type=str,
        #default="_output/target_lam_compare_ADR.png")
        #default="_output/target_lam_compare_30hpf.png")
        default="_output/target_lam_compare_dome.png")
        #default="_output/target_lam_compare_3day.png")
    parser.set_defaults()
    args = parser.parse_args(args)
    args.fishies = parse_comma_str(args.fishies, str)
    return args

def load_ADR_fish_params(fish):
    obs_file = "analyze_gestalt/_output/%s/sampling_seed0/fish_data_restrict.pkl" % fish
    with open(obs_file, "rb") as f:
        obs_dict = six.moves.cPickle.load(f)

    fitted_tree_file = "analyze_gestalt/_output/%s/sampling_seed0/sum_states_25/extra_steps_1/tune_pen_hanging.pkl" % fish
    with open(fitted_tree_file, "rb") as f:
        params = six.moves.cPickle.load(f)["final_fit"].model_params_dict

    return params["target_lams"], obs_dict

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
            targ_used[evt.min_target] += 1
    return targ_used

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
    return mean_corr, bootstrap_ci

def main(args=sys.argv[1:]):
    args = parse_args(args)
    fitted_params = {
            "mle": [],
            "simple": []}
    for fish in args.fishies:
        print(fish)
        if fish not in ["ADR1", "ADR2"]:
            # Load our estimated target rates
            file_name = os.path.join(args.folder, args.mle_file_template % fish)
            with open(file_name, "rb") as f:
                fitted_data = six.moves.cPickle.load(f)
                if "final_fit" in fitted_data:
                    res = fitted_data["final_fit"]
                else:
                    res = fitted_data[-1]["best_res"]
                fitted_param = res.model_params_dict["target_lams"]

            # Fit a simple target rate thing
            file_name = os.path.join(args.folder, args.obs_file_template % fish)
            with open(file_name, "rb") as f:
                obs_data_dict = six.moves.cPickle.load(f)
        else:
            fitted_param, obs_data_dict = load_ADR_fish_params(fish)

        fitted_params["mle"].append(fitted_param)
        fitted_params["simple"].append(_estimate_targ_rate_simple(obs_data_dict))

    for key in ["mle", "simple"]:
        print("fitting method", key)
        param_vals = fitted_params[key]
        all_corrs = _get_all_pairwise_correlations(param_vals)
        corr_mean, ci_95 = _get_summary_bootstrap(all_corrs, args.num_rands)
        print("Mean %.03f" % corr_mean)
        print("95 CI (%.03f, %.03f)" % (ci_95[0], ci_95[1]))

    # Plot all the params
    pd_data = []
    for param, fish in zip(fitted_params["mle"], args.fishies):
        for targ_idx, targ_val in enumerate(param):
            pd_data.append({
                "target": targ_idx,
                "val": targ_val,
                "fish": fish})
    data = pd.DataFrame(pd_data)
    sns.swarmplot(x="target", y="val", hue="fish", data=data)
    plt.xlabel("Target index")
    plt.ylabel("Estimated cut rate")
    plt.legend()
    plt.savefig(args.out_plot_file)

if __name__ == "__main__":
    main()
