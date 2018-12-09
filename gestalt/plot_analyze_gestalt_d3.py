"""
Converts the fitted tree to the JSON format requested
"""
import sys
import argparse
import json

from cell_lineage_tree import CellLineageTree
from plot_analyze_gestalt_meta import get_allele_to_cell_states, load_fish
import collapsed_tree

COLLAPSE_DIST = 0.001
ORGAN_TRANSLATION = {
    "7B_Brain": "Brain",
    "7B_Eye1": "Left eye",
    "7B_Eye2": "Right eye",
    "7B_Gills": "Gills",
    "7B_Intestine": "Intestinal bulb",
    "7B_Upper_GI": "Post intestine",
    "7B_Blood": "Blood",
    "7B_Heart_chunk": "Heart",
    "7B_Heart_diss": "DHC",
    "7B_Heart_GFP-": "NC",
    "7B_Heart_GFP+": "Cardiomyocytes",
}
ORGAN_COLORS = {
        "7B_Brain": "#4F6128",
        "7B_Eye1": "#77933C",
        "7B_Eye2": "#C3D69B",
        "7B_Gills": "#FFC000",
        "7B_Blood": "#FF0000",
        "7B_Heart_chunk": "#632523",
        "7B_Heart_GFP+": "#D99795",
        "7B_Heart_GFP-": "#E6B9B8",
        "7B_Heart_diss": "#943735",
        "7B_Upper_GI": "#558ED5",
        "7B_Intestine": "#8EB3E3",
}

def parse_args(args):
    parser = argparse.ArgumentParser(
            description="""
            make json format of gestalt tree for d3.
            """)
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
        default="_output/%s_tree.json")
    args = parser.parse_args(args)
    return args

def _expand_leaved_tree(fitted_bifurc_tree, allele_to_cell_state, cell_state_dict, default_dist_scale = 0, min_abund_thres = 0):
    """
    @param default_dist_scale: how much to assign the leaf branch to the different cell types vs. preserve as internal branch length
    @param min_abund_thres: minimum abundance for us to include that cell type leaf in the tree (we will always include
                        the cell type with the highest abundance, regardless of absolute abundance)
    """
    leaved_tree = fitted_bifurc_tree.copy()
    for l in leaved_tree:
        allele_str = l.allele_events_list_str
        if l.cell_state is None:
            old_dist = l.dist
            l.dist = old_dist * (1 - default_dist_scale)
            sorted_cell_states = sorted(
                    [(c_state_str, abund) for c_state_str, abund in allele_to_cell_state[allele_str].items()],
                    key = lambda c: c[1],
                    reverse=True)
            for c_state_str, abund in sorted_cell_states[:1]:
                new_child = CellLineageTree(
                    l.allele_list,
                    l.allele_events_list,
                    cell_state_dict[c_state_str],
                    dist = old_dist * default_dist_scale,
                    abundance = abund,
                    resolved_multifurcation = True)
                l.add_child(new_child)
            for c_state_str, abund in sorted_cell_states[1:]:
                if abund > min_abund_thres:
                    new_child = CellLineageTree(
                        l.allele_list,
                        l.allele_events_list,
                        cell_state_dict[c_state_str],
                        dist = old_dist * default_dist_scale,
                        abundance = abund,
                        resolved_multifurcation = True)
                    l.add_child(new_child)
    print("num leaves", len(leaved_tree))
    return leaved_tree

def convert_allele_events_to_event_str(leaf, bcode_meta):
    target_events = ["NONE"] * bcode_meta.n_targets
    for event in leaf.allele_events_list[0].events:
        min_deact_targ, max_deact_targ = event.get_min_max_deact_targets(bcode_meta)
        insert_repr = "%dI+%d+%s" % (event.insert_len, event.start_pos, event.insert_str)
        del_repr = "%dD+%d" % (event.del_len, event.start_pos)
        event_repr = del_repr if event.insert_len == 0 else "%s&%s" % (del_repr, insert_repr)
        for targ_idx in range(min_deact_targ, max_deact_targ + 1):
            target_events[targ_idx] = event_repr
    return "_".join(target_events)

def convert_to_json_recurse(
        curr_node,
        bcode_meta,
        organ_dict,
        organ_tot_counts,
        allele_to_cell_state,
        cell_state_dict):
    node_dict = {
            "name": str(curr_node.node_id),
            "parent": str(curr_node.up.node_id) if not curr_node.is_root() else "null",
            "length": curr_node.dist,
            "rootDist": curr_node.dist_to_root,
            "color": "black",
            "SAMPLE": "UNKNOWN",
            "justOrganSplit": False,
            "nodecolor": "black" if not curr_node.is_root() else "green",
            "organCountsMax": 0,
            "cladeTotal": 0,
            "max_organ_prop": 0.0,
            "event": "NONE_NONE_NONE_NONE_NONE_NONE_NONE_NONE_NONE_NONE",
            "commonEvent": "*_*_*_*_*_*_*_*_*_*",
            "organProportions": {"test": 1},
            "consistency": "NOTWT" if not curr_node.is_root() else "WT"
    }
    if curr_node.is_leaf():
        organ = organ_dict[str(curr_node.cell_state)]
        node_dict["sample"] = ORGAN_TRANSLATION[organ]
        node_dict["color"] = ORGAN_COLORS[organ]
        node_dict["event"] = convert_allele_events_to_event_str(curr_node, bcode_meta)
        organ_prop = curr_node.abundance/float(organ_tot_counts[organ])
        node_dict["organProportions"] = {node_dict["sample"]: organ_prop}
        node_dict["max_organ_prop"] = organ_prop
        print(node_dict)
        return node_dict

    node_dict["children"] = []
    for child in curr_node.children:
        node_dict["children"].append(
            convert_to_json_recurse(child, bcode_meta, organ_dict, organ_tot_counts, allele_to_cell_state, cell_state_dict)
        )
    return node_dict


def collapse_short_dists(fitted_bifurc_tree):
    # Collapse distances for plot readability
    for node in fitted_bifurc_tree.get_descendants():
        if node.dist < COLLAPSE_DIST:
            node.dist = 0
    col_tree = collapsed_tree.collapse_zero_lens(fitted_bifurc_tree)
    return col_tree

def make_organ_tot_counts(tree, organ_dict):
    print(organ_dict.keys())
    organ_tot_counts = {organ: 0 for organ in organ_dict.values()}
    for leaf in tree:
        organ = organ_dict[str(leaf.cell_state)]
        organ_tot_counts[organ] += leaf.abundance
    return organ_tot_counts

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    tree, obs_dict = load_fish(args.fish, args, method="PMLE", folder=args.folder)
    allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
    organ_dict = obs_dict["organ_dict"]
    bcode_meta = obs_dict["bcode_meta"]

    tree = collapse_short_dists(tree)
    tree = _expand_leaved_tree(tree, allele_to_cell_state, cell_state_dict)
    tree.label_node_ids()
    tree.label_dist_to_roots()
    organ_tot_counts = make_organ_tot_counts(tree, organ_dict)
    tree_json = convert_to_json_recurse(
        tree,
        bcode_meta,
        organ_dict,
        organ_tot_counts,
        allele_to_cell_state,
        cell_state_dict)

    with open(args.out_json % args.fish,"w") as out2:
        out2.write("[" + json.dumps(tree_json,sort_keys=False,indent=4) + "]\n")


if __name__ == "__main__":
    main()
