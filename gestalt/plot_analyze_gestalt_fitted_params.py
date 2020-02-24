"""
Print out the fitted params in a latex table
"""

import six
import sys
import argparse
import os.path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from bounded_distributions import ConditionalBoundedNegativeBinomial, ConditionalBoundedPoisson

from common import parse_comma_str, sigmoid

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='compare fitted params for different fish')
    parser.add_argument(
        '--obs-file-template',
        type=str,
        default="_output/%s/sampling_seed0/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-template',
        type=str,
        #default="_output/%s/sampling_seed0/sum_states_25/extra_steps_1/tune_pen_hanging.pkl")
        default="_output/%s/sampling_seed0/sum_states_20/extra_steps_1/tune_pen_hanging.pkl")
    parser.add_argument(
        '--fishies',
        type=str,
        #default="ADR1,ADR2,dome1,30hpf_v6_3,3day1")
        #default="ADR1,ADR2,dome1,3day1")
        default="30hpf_v6_4,30hpf_v6_5,30hpf_v6_6,30hpf_v6_8,dome1,dome3,dome8,dome10,3day1,3day2,3day3,3day4,3day5,3day6")
    parser.add_argument(
        '--out-table',
        type=str,
        default="_output/out.txt")
    parser.set_defaults()
    args = parser.parse_args(args)
    args.fishies = parse_comma_str(args.fishies, str)
    return args

def get_trim_zero_prob(all_model_params, is_left):
    index = 0 if is_left else 1
    return all_model_params["trim_zero_probs"][index * 2]

def get_insert_zero_prob(all_model_params):
    return all_model_params["insert_zero_prob"][0]

def get_insert_length_mean_sd(all_model_params):
    count = np.exp(all_model_params["insert_params"][0])
    prob = sigmoid(all_model_params["insert_params"][1])
    insert_dist = nbinom(count, 1 - prob)
    return insert_dist.mean() + 1, insert_dist.std()

def get_long_trim_length_mean(all_model_params, is_left):
    # TODO: take into account the inflation and stuff...
    start_idx = 0 if is_left else 2
    count = np.exp(all_model_params["trim_long_params"][start_idx + 0])
    prob = sigmoid(all_model_params["trim_long_params"][start_idx + 1])
    print(count)
    print(prob)
    return count * prob/(1 - prob)

def get_long_trim_length_sd(all_model_params, is_left):
    # TODO: take into account the inflation and stuff...
    start_idx = 0 if is_left else 2
    return 0

def get_short_trim_length_mean_sd(all_model_params, bcode_meta, is_left):
    start_idx = 0 if is_left else 2
    model_param_key = "trim_short_params"
    count = np.exp(all_model_params[model_param_key][start_idx + 0])
    logit = all_model_params[model_param_key][start_idx + 1]
    if is_left:
        trim_mins = 1
        trim_maxs = bcode_meta.left_long_trim_min
    else:
        trim_mins = 1
        trim_maxs = bcode_meta.right_long_trim_min
    mean_long_max = int(np.mean(trim_maxs))
    mean_long_min = int(np.mean(trim_mins))
    trim_dist = ConditionalBoundedNegativeBinomial(mean_long_min, mean_long_max, count, logit)

    # Now get all the moments
    tot_pmf = 0
    first_moment = 0
    second_moment = 0
    for i in range(mean_long_min, mean_long_max + 1):
        first_moment += i * trim_dist.pmf(i)
        second_moment += np.power(i, 2) * trim_dist.pmf(i)
        tot_pmf += trim_dist.pmf(i)
    assert np.isclose(tot_pmf, 1)
    std_dev = second_moment - np.power(first_moment, 2)
    return first_moment, np.sqrt(std_dev)

def get_long_trim_length_mean_sd(all_model_params, bcode_meta, is_left):
    model_param_key = "trim_long_params"
    est_mean = np.exp(all_model_params[model_param_key][is_left])
    if is_left:
        trim_mins = bcode_meta.left_long_trim_min
        trim_maxs = bcode_meta.left_max_trim
    else:
        trim_mins = bcode_meta.right_long_trim_min
        trim_maxs = bcode_meta.right_max_trim
    mean_long_max = int(np.mean(trim_maxs))
    mean_long_min = int(np.mean(trim_mins))
    trim_dist = ConditionalBoundedPoisson(mean_long_min, mean_long_max, est_mean)

    # Now get all the moments
    tot_pmf = 0
    first_moment = 0
    second_moment = 0
    for i in range(mean_long_min, mean_long_max + 1):
        first_moment += i * trim_dist.pmf(i)
        second_moment += np.power(i, 2) * trim_dist.pmf(i)
        tot_pmf += trim_dist.pmf(i)
    assert np.isclose(tot_pmf, 1)
    std_dev = second_moment - np.power(first_moment, 2)
    return first_moment, np.sqrt(std_dev)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    fishes_found = []
    fish_param_dicts = []
    for fish in args.fishies:
        print(fish)
        # Load our estimated target rates
        file_name = args.mle_template % fish
        try:
            with open(file_name, "rb") as f:
                fitted_data = six.moves.cPickle.load(f)
                if "final_fit" in fitted_data:
                    res = fitted_data["final_fit"]
                else:
                    res = fitted_data[-1]["best_res"]
                all_model_params = res.model_params_dict
        except Exception as e:
            print("file issue", e)
            continue

        # Fit a simple target rate thing
        file_name = args.obs_file_template % fish
        with open(file_name, "rb") as f:
            obs_data_dict = six.moves.cPickle.load(f)
        bcode_meta = obs_data_dict["bcode_meta"]

        sorted_keys = sorted(list(all_model_params.keys()))
        for k in sorted_keys:
            if k not in ["branch_len_offsets_proportion", "branch_len_inners"]:
                v = all_model_params[k]
                print(k, ":", v)
        fish_param_dict = {}
        for idx, target_lam in enumerate(all_model_params["target_lams"]):
            fish_param_dict["Target%d" % (idx + 1)] = target_lam
        fishes_found.append(fish)
        fish_param_dict["Double cut rate"] = all_model_params["double_cut_weight"][0]
        fish_param_dict["Long factor left"] = all_model_params["trim_long_factor"][0]
        fish_param_dict["Long factor right"] = all_model_params["trim_long_factor"][1]
        mean, sd = get_short_trim_length_mean_sd(all_model_params, bcode_meta, is_left=1)
        fish_param_dict["Left short trim length mean"] = mean
        fish_param_dict["Left short trim length sd"] = sd
        mean, sd = get_short_trim_length_mean_sd(all_model_params, bcode_meta, is_left=0)
        fish_param_dict["Right short trim length mean"] = mean
        fish_param_dict["Right short trim length sd"] = sd
        mean, sd = get_long_trim_length_mean_sd(all_model_params, bcode_meta, is_left=1)
        fish_param_dict["Left long trim length mean"] = mean
        fish_param_dict["Left long trim length sd"] = sd
        mean, sd = get_long_trim_length_mean_sd(all_model_params, bcode_meta, is_left=0)
        fish_param_dict["Right long trim length mean"] = mean
        fish_param_dict["Right long trim length sd"] = sd
        fish_param_dict["Left short trim zero prob"] = get_trim_zero_prob(all_model_params, is_left=1)
        fish_param_dict["Right short trim zero prob"] = get_trim_zero_prob(all_model_params, is_left=0)
        fish_param_dict["Insertion zero prob"] = get_insert_zero_prob(all_model_params)
        mean, sd = get_insert_length_mean_sd(all_model_params)
        fish_param_dict["Insertion length mean"] = mean
        fish_param_dict["Insertion length sd"] = sd
        fish_param_dicts.append(fish_param_dict)

    df = pd.DataFrame(fish_param_dicts, index=fishes_found)
    latex_str = df.transpose().to_latex(float_format=lambda x: '%.3f' % x)
    print(latex_str)
    with open(args.out_table, "w") as f:
        f.write(latex_str)

if __name__ == "__main__":
    main()
