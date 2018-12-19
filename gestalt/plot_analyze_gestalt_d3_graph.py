"""
Converts the fitted tree to the JSON format requested
"""
import sys
import argparse
import json

from cell_lineage_tree import CellLineageTree
from plot_analyze_gestalt_meta import get_allele_to_cell_states, load_fish
from plot_analyze_gestalt_d3 import _expand_leaved_tree, convert_allele_events_to_event_str, make_organ_tot_counts, collapse_short_dists
from plot_analyze_gestalt_cell_type_times import HEART_TYPES, EYE_TYPES, GUT_TYPES, ORGAN_TYPES
import collapsed_tree

def parse_args(args):
    parser = argparse.ArgumentParser(
            description="""
            make json format of gestalt tree for d3.
            """)
    parser.add_argument(
        '--num-discretize',
        type=int,
        default=10)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/sum_states_20/extra_steps_1/tune_pen_hanging.pkl")
    parser.add_argument(
        '--chronos-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/chronos_fitted.pkl")
    parser.add_argument(
        '--nj-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/nj_fitted.pkl")
    parser.add_argument(
        '--fish',
        type=str,
        default="ADR1")
    parser.add_argument(
        '--folder',
        type=str,
        default="")
    parser.add_argument(
        '--out-json',
        type=str,
        default="_output/%s_%s_graph_d3.json")
    args = parser.parse_args(args)
    return args

def discretize_tree_as_graph(
        graph_links,
        graph_nodes,
        start_clt,
        curr_clt,
        curr_idx,
        num_discretize=10):
    def _create_link(source_clt, target_clt, source_node_id, target_node_id):
        link_id = (source_node_id, target_node_id)
        if link_id not in graph_links:
            graph_links[link_id] = set()
        graph_links[link_id].add(source_clt.node_id)
        for node_id, node_clt in [(source_node_id, source_clt), (target_node_id, target_clt)]:
            if node_id not in graph_nodes:
                graph_nodes[node_id] = set()
            graph_nodes[node_id].add(node_clt.node_id)

    end_idx = int(curr_clt.dist_to_root * num_discretize)
    if curr_idx == end_idx - 1:
        source_node_id = (start_clt.cell_type_str, curr_idx)
        target_node_id = (curr_clt.cell_type_str, curr_idx + 1)
        _create_link(start_clt, curr_clt, source_node_id, target_node_id)
        new_clt = curr_clt
    elif curr_idx < end_idx - 1:
        halfway_idx = curr_idx + int((end_idx - curr_idx)/2)
        for i in range(curr_idx, halfway_idx):
            source_node_id = (start_clt.cell_type_str, i)
            target_node_id = (start_clt.cell_type_str, i + 1)
            _create_link(start_clt, curr_clt, source_node_id, target_node_id)
        source_node_id = (start_clt.cell_type_str, halfway_idx)
        target_node_id = (curr_clt.cell_type_str, halfway_idx + 1)
        _create_link(start_clt, curr_clt, source_node_id, target_node_id)
        for i in range(halfway_idx + 1, end_idx):
            source_node_id = (curr_clt.cell_type_str, i)
            target_node_id = (curr_clt.cell_type_str, i + 1)
            _create_link(curr_clt, curr_clt, source_node_id, target_node_id)
        new_clt = curr_clt
    else:
        new_clt = start_clt

    for child in curr_clt.children:
        discretize_tree_as_graph(
                graph_links,
                graph_nodes,
                new_clt,
                child,
                end_idx,
                num_discretize)

def label_tree_cell_types(fitted_bifurc_tree, organ_dict, allele_to_cell_state):
    for node in fitted_bifurc_tree.traverse('postorder'):
        node.add_feature("cell_types", {tissue: 0 for tissue in ORGAN_TYPES})
        if node.is_leaf():
            allele_str = node.allele_events_list_str
            cell_types = allele_to_cell_state[allele_str]
            for tissue_key, abundance in cell_types.items():
                tissue_str = organ_dict[tissue_key].replace("7B_", "")
                if tissue_str in set(HEART_TYPES):
                    node.cell_types["Heart"] += abundance
                elif tissue_str in set(GUT_TYPES):
                    node.cell_types["Int"] += abundance
                elif tissue_str in set(EYE_TYPES):
                    node.cell_types["Eyes"] += abundance
                elif tissue_str == "Blood":
                    continue
                else:
                    node.cell_types[tissue_str] = abundance
        else:
            for child in node.children:
                for key, abundance in child.cell_types.items():
                    node.cell_types[key] += abundance

    for node in fitted_bifurc_tree.traverse():
        node.add_feature(
            "cell_type_str",
            "".join(list(sorted([c[0] for c, abund in node.cell_types.items() if abund > 0]))))

def prune_tree(tree, filter_organ):
    keep_ids = set()
    for leaf in tree:
        cell_types = [c for c, v in leaf.cell_types.items() if v > 0]
        if len(cell_types) == 1:
            if filter_organ in cell_types:
                keep_ids.add(leaf.node_id)
    return CellLineageTree.prune_tree(tree, keep_ids)

def create_json(tree, graph_links, graph_nodes):
    sorted_tissues = sorted(ORGAN_TYPES)
    tissue_proportions = {}
    for node in tree.traverse():
        tissue_proportions[node.node_id] = {
            tissue: abundance for tissue, abundance in node.cell_types.items()}

    sorted_node_keys = sorted(set([k[0] for k in graph_nodes.keys()]), key=lambda x: len(x))
    graph_yPos = {key: idx for idx, key in enumerate(sorted_node_keys)}

    node_dict = {}
    graph_node_json = []
    for node_idx, (node_id, clt_node_ids) in enumerate(graph_nodes.items()):
        node_dict[node_id] = node_idx
        cell_type_str = node_id[0]
        time_idx = node_id[1]
        tissue_dict = {tissue_str: 0 for tissue_str in ORGAN_TYPES}
        for tissue_str in ORGAN_TYPES:
            for clt_node_id in clt_node_ids:
                tissue_dict[tissue_str] += tissue_proportions[clt_node_id][tissue_str]
        tot_abundance = sum(tissue_dict.values())
        graph_node_tissue_proportions = [
            tissue_dict[tissue]/tot_abundance for tissue in sorted_tissues]
        graph_node_json.append({
            "node": node_idx,
            "name": "%s_%d" % node_id,
            "tissues": graph_node_tissue_proportions,
            "yPos": graph_yPos[cell_type_str],
            "time_idx": time_idx,
            "num_uniqs": len(clt_node_ids)})

    graph_links_json = []
    for link_key, link_clt_node_ids in graph_links.items():
        source_id = node_dict[link_key[0]]
        target_id = node_dict[link_key[1]]
        graph_links_json.append({
            "source": source_id,
            "target": target_id,
            "value": len(link_clt_node_ids)})
    return {
        "tissues": sorted_tissues,
        "nodes": graph_node_json,
        "links": graph_links_json}

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    tree, obs_dict = load_fish(args.fish, args, method="PMLE", folder=args.folder)
    allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
    organ_dict = obs_dict["organ_dict"]

    tree = collapse_short_dists(tree)
    tree = _expand_leaved_tree(tree, allele_to_cell_state, cell_state_dict)
    tree.label_node_ids()
    tree.label_dist_to_roots()
    label_tree_cell_types(tree, organ_dict, allele_to_cell_state)
    #organ_tot_counts = make_organ_tot_counts(tree, organ_dict)

    for filter_organ in ORGAN_TYPES:
        print("filter organ", filter_organ)
        pruned_tree = prune_tree(tree, filter_organ)
        graph_links = {}
        graph_nodes = {}
        discretize_tree_as_graph(
                graph_links,
                graph_nodes,
                pruned_tree,
                pruned_tree,
                0,
                num_discretize=args.num_discretize)
        graph_json = create_json(tree, graph_links, graph_nodes)

        with open(args.out_json % (args.fish, filter_organ),"w") as out2:
            out2.write(json.dumps(graph_json, indent=4))


if __name__ == "__main__":
    main()
