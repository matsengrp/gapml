"""
Converts the fitted tree to the JSON format requested
"""
import sys
import argparse
import json

from plot_analyze_gestalt_meta import get_allele_to_cell_states, load_fish
from plot_analyze_gestalt_tree import _expand_leaved_tree
import collapsed_tree

COLLAPSE_DIST = 0.001

def parse_args(args):
    parser = argparse.ArgumentParser(
            description="""
            make json format of gestalt tree for d3.
            """)
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
            "nodecolor": "black",
            "organCountsMax": 0,
            "cladeTotal": 0,
            "max_organ_prop": 0.0,
            "event": "NONE_NONE_NONE_NONE_NONE_NONE_NONE_NONE_NONE_NONE",
            "commonEvent": "*_*_*_*_*_*_*_*_*_*",
            "organProportions": {"test": 1},
            "consistency": "NOTWT" if not curr_node.is_root() else "WT"
    }
    if curr_node.is_leaf():
        node_dict["sample"] = organ_dict[str(curr_node.cell_state)]
        node_dict["event"] = convert_allele_events_to_event_str(curr_node, bcode_meta)
        node_dict["organProportions"] = {node_dict["sample"]: 0.001}
        node_dict["max_organ_prop"] = 0.0001
        print(node_dict)
        return node_dict

    node_dict["children"] = []
    for child in curr_node.children:
        node_dict["children"].append(
            convert_to_json_recurse(child, bcode_meta, organ_dict, allele_to_cell_state, cell_state_dict)
        )
    return node_dict


def collapse_short_dists(fitted_bifurc_tree):
    # Collapse distances for plot readability
    for node in fitted_bifurc_tree.get_descendants():
        if node.dist < COLLAPSE_DIST:
            node.dist = 0
    col_tree = collapsed_tree.collapse_zero_lens(fitted_bifurc_tree)
    return col_tree

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    # TODO: this doesnt work right now. need to add in prefix of tmp_mount
    tree, obs_dict = load_fish(args.fish, method="PMLE", folder=args.folder)
    allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
    organ_dict = obs_dict["organ_dict"]
    bcode_meta = obs_dict["bcode_meta"]

    tree = collapse_short_dists(tree)
    tree = _expand_leaved_tree(tree, allele_to_cell_state, cell_state_dict)
    tree.label_node_ids()
    tree.label_dist_to_roots()
    tree_json = convert_to_json_recurse(
        tree,
        bcode_meta,
        organ_dict,
        allele_to_cell_state,
        cell_state_dict)

    with open(args.out_json % args.fish,"w") as out2:
        out2.write("[" + json.dumps(tree_json,sort_keys=False,indent=4) + "]\n")


if __name__ == "__main__":
    main()
