import six
import sys
import argparse
import os.path
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from common import parse_comma_str, sigmoid

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='compare fitted params for different fish')
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
        '--fishies',
        type=str,
        default="ADR1,ADR2,dome1,30hpf_v6_5,3day1")
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

    return params, obs_dict

def get_insert_zero_prob(all_model_params):
    # TODO: take into account the inflation and stuff...
    return -1

def get_insert_length_mean(all_model_params):
    count = np.exp(all_model_params["insert_params"][0])
    prob = sigmoid(all_model_params["insert_params"][1])
    insert_dist = nbinom(count, 1 - prob)
    return insert_dist.mean() + 1

def get_insert_length_sd(all_model_params):
    count = np.exp(all_model_params["insert_params"][0])
    prob = sigmoid(all_model_params["insert_params"][1])
    insert_dist = nbinom(count, 1 - prob)
    return insert_dist.std()

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

def get_short_trim_length_mean(all_model_params, is_left):
    # TODO: take into account the inflation and stuff...
    start_idx = 0 if is_left else 2
    count = np.exp(all_model_params["trim_short_params"][start_idx + 0])
    prob = sigmoid(all_model_params["trim_short_params"][start_idx + 1])
    print("coun", count, "prob", prob)
    trim_dist = nbinom(count, 1 - prob)
    return trim_dist.mean() + 1

def get_short_trim_length_sd(all_model_params, is_left):
    # TODO: take into account the inflation and stuff...
    start_idx = 0 if is_left else 2
    return 0

def main(args=sys.argv[1:]):
    args = parse_args(args)
    fish_param_dicts = []
    for fish in args.fishies:
        print(fish)
        if fish not in ["ADR1", "ADR2"]:
            # Load our estimated target rates
            file_name = os.path.join(args.folder, args.mle_file_template % fish)
            with open(file_name, "rb") as f:
                res = six.moves.cPickle.load(f)["final_fit"]
                all_model_params = res.model_params_dict

            # Fit a simple target rate thing
            file_name = os.path.join(args.folder, args.obs_file_template % fish)
            with open(file_name, "rb") as f:
                obs_data_dict = six.moves.cPickle.load(f)
        else:
            all_model_params, obs_data_dict = load_ADR_fish_params(fish)
        sorted_keys = sorted(list(all_model_params.keys()))
        for k in sorted_keys:
            if k not in ["branch_len_offsets_proportion", "branch_len_inners", "boost_probs"]:
                v = all_model_params[k]
                print(k, ":", v)
        fish_param_dict = {}
        for idx, target_lam in enumerate(all_model_params["target_lams"]):
            fish_param_dict["Target%d" % (idx + 1)] = target_lam
        fish_param_dict["Double cut rate"] = all_model_params["double_cut_weight"][0]
        fish_param_dict["Long factor left"] = all_model_params["trim_long_factor"][0]
        fish_param_dict["Long factor right"] = all_model_params["trim_long_factor"][1]
        fish_param_dict["Left short trim length mean"] = get_short_trim_length_mean(all_model_params, is_left=True)
        fish_param_dict["Left short trim length SD"] = get_short_trim_length_sd(all_model_params, is_left=True)
        fish_param_dict["Right short trim length mean"] = get_short_trim_length_mean(all_model_params, is_left=False)
        fish_param_dict["Right short trim length SD"] = get_short_trim_length_sd(all_model_params, is_left=False)
        fish_param_dict["Left long trim length mean"] = get_long_trim_length_mean(all_model_params, is_left=True)
        fish_param_dict["Left long trim length SD"] = get_long_trim_length_sd(all_model_params, is_left=True)
        fish_param_dict["Right long trim length mean"] = get_long_trim_length_mean(all_model_params, is_left=False)
        fish_param_dict["Right long trim length SD"] = get_long_trim_length_sd(all_model_params, is_left=False)
        fish_param_dict["Insertion zero prob"] = get_insert_zero_prob(all_model_params)
        fish_param_dict["Insertion length mean"] = get_insert_length_mean(all_model_params)
        fish_param_dict["Insertion length SD"] = get_insert_length_sd(all_model_params)
        fish_param_dicts.append(fish_param_dict)

    df = pd.DataFrame(fish_param_dicts, index=args.fishies)
    print(df.transpose().to_latex(float_format=lambda x: '%.3f' % x))

if __name__ == "__main__":
    main()
