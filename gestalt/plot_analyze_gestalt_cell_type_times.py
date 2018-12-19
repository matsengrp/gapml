import sys
import argparse
import subprocess
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from plot_analyze_gestalt_meta import load_fish, get_allele_to_cell_states

"""
Plot distribution of (differentiated and pluripotent) cell type times
"""

MIN_OBS = 5
EYE_TYPES = [
    "Eye1",
    "Eye2",
]
GUT_TYPES = [
    "Upper_GI",
    "Intestine",
]
HEART_TYPES = [
    "Heart_GFP+",
    "Heart_chunk",
    "Heart_GFP-",
    "Heart_diss",
]

ORGAN_TYPES = [
    "Brain",
    "Eyes",
    "Gills",
    "Heart",
    "Int",
]
def parse_args(args):
    parser = argparse.ArgumentParser(
            description="""
            Plot alluvial in ggplot
            """)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/sum_states_20/extra_steps_1/tune_pen_hanging.pkl")
    args = parser.parse_args(args)
    return args

def label_tree_cell_types(fitted_bifurc_tree, organ_dict, allele_to_cell_state):
    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            allele_str = node.allele_events_list_str
            cell_types = allele_to_cell_state[allele_str]
            node.add_feature(
                "cell_types",
                set([organ_dict[x].replace("7B_", "") for x in cell_types.keys()])
            )
            if node.cell_types.intersection(set(HEART_TYPES)):
                node.cell_types = node.cell_types - set(HEART_TYPES)
                node.cell_types.add("Heart")
            if node.cell_types.intersection(set(GUT_TYPES)):
                node.cell_types = node.cell_types - set(GUT_TYPES)
                node.cell_types.add("Int")
            if node.cell_types.intersection(set(EYE_TYPES)):
                node.cell_types = node.cell_types - set(EYE_TYPES)
                node.cell_types.add("Eyes")
            # if see if render better
            node.cell_types = node.cell_types - set(["Blood"])
        else:
            node.add_feature("cell_types", set())
            for child in node.children:
                node.cell_types.update(child.cell_types)

def make_cell_type_str(cell_types):
    return "".join([c[0] for c in sorted(list(cell_types))])

def get_cell_type_divergence_times(
        fitted_bifurc_tree,
        organ_dict,
        allele_to_cell_state,
        filter_organ=None,
        time_indices=[0,1,2,3,4]):
    label_tree_cell_types(fitted_bifurc_tree, organ_dict, allele_to_cell_state)

    cell_type_flow_df = []
    total_abund = 0
    for leaf in fitted_bifurc_tree:
        if len(leaf.cell_types) > 1 or (filter_organ is not None and filter_organ not in leaf.cell_types):
            continue

        cell_types = allele_to_cell_state[leaf.allele_events_list_str]
        leaf_abund = list(cell_types.values())[0]
        total_abund += leaf_abund
        leaf_organ = make_cell_type_str(leaf.cell_types)
        cell_type_flow_df.append(
                [leaf_organ, len(time_indices), leaf.node_id, leaf_abund])

        node = leaf
        time_interval = 1./len(time_indices)
        discrete_time_index = len(time_indices) - 1
        while not node.is_root():
            if discrete_time_index == 0:
                node = node.up
                continue

            discrete_time = discrete_time_index * time_interval
            if node.dist_to_root > discrete_time and node.up.dist_to_root < discrete_time:
                cell_types = list(node.cell_types)
                cell_type_str = make_cell_type_str(cell_types)
                cell_type_flow_df.append(
                        [cell_type_str, discrete_time_index, leaf.node_id, leaf_abund])
                discrete_time_index -= 1
            else:
                node = node.up

        root_cell_type_str = make_cell_type_str(node.cell_types)
        cell_type_flow_df.append(
                [root_cell_type_str, 0, leaf.node_id, leaf_abund])

    flow_df = pd.DataFrame(cell_type_flow_df, columns=['progenitor', 'time', 'leaf_id', 'Freq'])
    flow_df["Freq"] /= total_abund
    return flow_df

def plot_flow(flow_df, out_plot):
    if flow_df.shape[0] == 0:
        print("NOTHING TO PLOT")
        return

    in_file = "_output/alluvial.csv"
    flow_df.to_csv(in_file, index=False)

    cmd = [
            'Rscript',
            '../R/plot_alluvial.R',
            in_file,
            out_plot,
    ]

    print("Calling:", " ".join(cmd))
    res = subprocess.call(cmd)
    assert res == 0
    print("resss", res)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    sns.set_context('poster')
    fishies = ["ADR1", "ADR2"]
    methods = ["PMLE"]
    all_dfs = {}
    for method in methods:
        for fish_idx, fish in enumerate(fishies):
            tree, obs_dict = load_fish(fish, args, method)
            tree.label_dist_to_roots()
            organ_dict = obs_dict["organ_dict"]
            allele_to_cell_state, _ = get_allele_to_cell_states(obs_dict)

            for cell_type in ORGAN_TYPES:
                cell_type_time_df = get_cell_type_divergence_times(
                    tree,
                    organ_dict,
                    allele_to_cell_state,
                    filter_organ=cell_type,
                    time_indices=list(range(20)))
                cell_type_time_df["fish"] = fish
                cell_type_time_df["cell_type"] = cell_type

                if cell_type not in all_dfs:
                    all_dfs[cell_type] = []
                all_dfs[cell_type].append(cell_type_time_df)

        for cell_type in ORGAN_TYPES:
            out_plot_template = "_output/cell_type_divergence_times_%s_%s.png" % (cell_type, method)
            print(out_plot_template)
            plot_flow(pd.concat(all_dfs[cell_type], ignore_index=True), out_plot_template)

if __name__ == "__main__":
    main()
