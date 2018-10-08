import six
import sys
import argparse
import os.path
import numpy as np
import scipy.stats

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from common import parse_comma_str

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='tune over topologies and fit model parameters')
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
        default="tmp_mount/analyze_gestalt")
    parser.add_argument(
        '--fishies',
        type=str,
        default="30hpf1_abund2,30hpf2_abund1,30hpf3_abund2,30hpf4_abund1,30hpf5_abund1")
    parser.add_argument(
        '--out-plot-file',
        type=str,
        default="_output/target_lam_compare.png")
    parser.set_defaults()
    args = parser.parse_args(args)
    args.fishies = parse_comma_str(args.fishies, str)
    return args

def _get_normalized_params(param):
    return param/np.sum(param)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    fitted_params = {
            "mle": [],
            "simple": []}
    simple_fitted_params = []
    for fish in args.fishies:
        print(fish)
        # Load our estimated target rates
        file_name = os.path.join(args.folder, args.mle_file_template % fish)
        with open(file_name, "rb") as f:
            res = six.moves.cPickle.load(f)["final_fit"]
            fitted_param = res.model_params_dict["target_lams"]
            fitted_params["mle"].append(
                    _get_normalized_params(fitted_param))

        # Fit a simple target rate thing
        file_name = os.path.join(args.folder, args.obs_file_template % fish)
        with open(file_name, "rb") as f:
            obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]
        obs_data = obs_data_dict["obs_leaves"]
        assert bcode_meta.num_barcodes == 1

        all_events = [o.allele_events_list[0].events for o in obs_data]
        uniq_events = set([evt for evts in all_events for evt in evts])
        targ_used = np.zeros(bcode_meta.n_targets)
        for evt in uniq_events:
            if evt.min_target != evt.max_target:
                targ_used[evt.min_target] += 1
                targ_used[evt.max_target] += 1
            else:
                targ_used[evt.min_target] += 1
        fitted_params["simple"].append(_get_normalized_params(targ_used))

    
    for key, param_vals in fitted_params.items():
        all_corrs = []
        for idx1, fit_param1 in enumerate(param_vals):
            for fit_param2 in param_vals[idx1 + 1:]:
                #corr = scipy.stats.pearsonr(fit_param1, fit_param2)[0]
                corr = scipy.stats.spearmanr(fit_param1, fit_param2)[0]
                #corr = np.linalg.norm(np.array(fit_param1) - np.array(fit_param2), ord=1)
                all_corrs.append(corr)
        corr_mean = np.mean(all_corrs)
        std_err = np.sqrt(np.var(all_corrs)/len(all_corrs))
        print("Mean", corr_mean)
        print("SE", std_err)
        print("95 CI", corr_mean - 1.96 * std_err, corr_mean + 1.96 * std_err)

    # Plot all the params
    for param, fish in zip(fitted_params["mle"], args.fishies):
        plt.plot(param, label=fish)
    plt.xlabel("Target index")
    plt.ylabel("Estimated cut rate")
    plt.legend()
    plt.savefig(args.out_plot_file)

if __name__ == "__main__":
    main()
