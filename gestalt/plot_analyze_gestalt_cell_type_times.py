import sys
import subprocess
import matplotlib
matplotlib.use('Agg')
import os.path
import seaborn as sns
import numpy as np
import six
import scipy.stats
from cell_lineage_tree import CellLineageTree
from matplotlib import pyplot as plt
import pandas as pd
from scipy.ndimage import convolve1d

from plot_analyze_gestalt_meta import load_fish, get_allele_to_cell_states, ORGAN_LABELS

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
    "Gut",
    "Gills",
    "Blood",
    "Heart",
]

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
                node.cell_types.add("Gut")
            if node.cell_types.intersection(set(EYE_TYPES)):
                node.cell_types = node.cell_types - set(EYE_TYPES)
                node.cell_types.add("Eyes")
        else:
            node.add_feature("cell_types", set())
            for child in node.children:
                node.cell_types.update(child.cell_types)

def get_cell_type_times(fitted_bifurc_tree, organ_dict, allele_to_cell_state):
    label_tree_cell_types(fitted_bifurc_tree, organ_dict, allele_to_cell_state)

    cell_type_times = {
            "organ": [],
            "time": []}
    # Keep track of how many unique branches of each category were found
    cell_type_counter = {}
    for node in fitted_bifurc_tree.traverse():
        if node.is_root():
            continue

        if node.is_leaf() and len(allele_to_cell_state[node.allele_events_list_str]) > 1:
            continue

        # ensure that cell types are still able to differentiate eventually
        # (otherwise there is a major barcode exhaustion issue)
        if not node.is_leaf() and not any([len(allele_to_cell_state[leaf.allele_events_list_str]) == 1 for leaf in node]):
            continue

        if not node.is_leaf() and len(node.spine_children) == 0 and any([len(node.cell_types - c.cell_types) > 0 for c in node.children]):
            continue

        cell_type_abund_dict = node.cell_types
        cell_types = list(cell_type_abund_dict)

        cell_type_str = "+".join(list(sorted(cell_types)))
        time_incr = 0.02
        start_time = node.up.dist_to_root
        end_time = node.dist_to_root
        for t_grid in np.arange(start_time, end_time, time_incr):
            cell_type_times["organ"].append(cell_type_str)
            # Get the time when this node that only depends to this cell type group branched off
            cell_type_times["time"].append(t_grid)
        if node.is_leaf():
            for t_grid in np.arange(1, 1.1, time_incr):
                # HACK: To make the violin plots of singleton cell types look less confusing
                cell_type_times["organ"].append(cell_type_str)
                cell_type_times["time"].append(t_grid)

        if cell_type_str not in cell_type_counter:
            cell_type_counter[cell_type_str] = 1
        else:
            cell_type_counter[cell_type_str] += 1

    # we're only interested in categories with enough branches
    for k, v in cell_type_counter.items():
        print(k, v)
    many_obs_cell_types = [k for k, count in cell_type_counter.items() if count >= MIN_OBS]
    cell_type_times_df = pd.DataFrame(cell_type_times)
    return cell_type_times_df[cell_type_times_df["organ"].isin(many_obs_cell_types)]

def plot_violins(cell_type_times, out_plot_file):
    # Sort organ grouping by time
    organ_times = cell_type_times.groupby(['organ'])['time'].min()
    print(organ_times.index[np.argsort(organ_times)])
    organ_order = organ_times.index[np.argsort(organ_times)]

    plt.clf()
    plt.figure(figsize=(15,30))
    sns.set(font_scale=3)
    sns.violinplot(
            x="time",
            y="organ",
            order=organ_order,
            orient='h',
            scale="count",
            cut=0.05,
            data=cell_type_times)
    #sns.swarmplot(
    #        x="time",
    #        y="organ",
    #        order=organ_order,
    #        orient='h',
    #        data=cell_type_times,
    #        color="white",
    #        edgecolor="gray")
    plt.xlim(0, 1)
    plt.savefig(out_plot_file, transparent=True, bbox_inches='tight')
    print("matrix PLOT", out_plot_file)

def get_cell_type_divergence_times(
        fitted_bifurc_tree,
        organ_dict,
        allele_to_cell_state,
        filter_organ=None,
        time_indices=[0,1,2,3,4]):
    label_tree_cell_types(fitted_bifurc_tree, organ_dict, allele_to_cell_state)

    cell_type_flow_df = []
    for leaf in fitted_bifurc_tree:
        if len(leaf.cell_types) > 1 or (filter_organ is not None and filter_organ not in leaf.cell_types):
            continue

        cell_types = allele_to_cell_state[leaf.allele_events_list_str]
        leaf_abund = list(cell_types.values())[0]
        leaf_organ = list(leaf.cell_types)[0]
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
                cell_type_str = "+".join(list(sorted(cell_types)))
                cell_type_flow_df.append(
                        [cell_type_str, discrete_time_index, leaf.node_id, leaf_abund])
                discrete_time_index -= 1
            else:
                node = node.up

        root_cell_type_str = "+".join(list(sorted(list(node.cell_types))))
        cell_type_flow_df.append(
                [root_cell_type_str, 0, leaf.node_id, leaf_abund])

    flow_df = pd.DataFrame(cell_type_flow_df, columns=['progenitor', 'time', 'leaf_id', 'Freq'])
    return flow_df

def plot_flow(flow_df, out_plot):
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
    print("resss", res)

def main(args=sys.argv[1:]):
    sns.set_context('poster')
    fishies = ["ADR1", "ADR2"]
    methods = ["PMLE"]
    for method in methods:
        for fish_idx, fish in enumerate(fishies):
            tree, obs_dict = load_fish(fish, method)
            tree.label_dist_to_roots()
            organ_dict = obs_dict["organ_dict"]
            allele_to_cell_state, _ = get_allele_to_cell_states(obs_dict)
            #cell_type_times = get_cell_type_times(tree, organ_dict, allele_to_cell_state)
            #out_plot_file = "_output/cell_type_times_%s_%s.png" % (fish, method)
            #plot_violins(cell_type_times, out_plot_file)

            for cell_type in ORGAN_TYPES:
                cell_type_time_df = get_cell_type_divergence_times(
                    tree,
                    organ_dict,
                    allele_to_cell_state,
                    filter_organ=cell_type)
                out_plot_template = "_output/cell_type_divergence_times_%s_%s.png" % (fish, cell_type)
                print(out_plot_template)
                plot_flow(cell_type_time_df, out_plot_template)

if __name__ == "__main__":
    main()
