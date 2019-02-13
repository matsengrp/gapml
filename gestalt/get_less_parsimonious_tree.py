import sys
import argparse
import random
import six
import numpy as np

import ancestral_events_finder
from barcode_metadata import BarcodeMetadata
from cell_lineage_tree import CellLineageTree


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='make a less parsimonious tree')
    parser.add_argument(
        '--seed',
        type=int,
        default=40)
    parser.add_argument(
        '--num-spr-moves',
        type=int,
        default=1)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data_b1.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--parsimony-topology-file',
        type=str,
        default="_output/parsimony_tree0.pkl",
        help="Topology file")
    parser.add_argument(
        '--out-tree-file',
        type=str,
        default="_output/less_parsimonious_tree.pkl")
    parser.set_defaults()
    args = parser.parse_args(args)
    return args

def get_tree_parsimony_score(tree: CellLineageTree, bcode_meta: BarcodeMetadata):
    """
    @return the parsimony score for the given tree
    """
    ancestral_events_finder.annotate_ancestral_states(tree, bcode_meta)
    return ancestral_events_finder.get_parsimony_score(tree)

def do_spr_move(tree: CellLineageTree, bcode_meta: BarcodeMetadata):
    num_leaves = len(tree)
    max_node_id = tree.label_node_ids()
    orig_pars_score = get_tree_parsimony_score(tree, bcode_meta)
    while True:
        max_node_id = tree.label_node_ids()
        rand_node_source, rand_node_target = np.random.choice(np.arange(2, max_node_id), size=2, replace=False)
        source_node = tree.search_nodes(node_id=rand_node_source)[0]
        target_node = tree.search_nodes(node_id=rand_node_target)[0]
        source_node.detach()
        target_node.up.add_child(source_node)
        assert len(tree) == num_leaves
        new_pars_score = get_tree_parsimony_score(tree, bcode_meta)
        if new_pars_score != orig_pars_score:
            break

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.obs_file, "rb") as f:
        obs_data = six.moves.cPickle.load(f)
        bcode_meta = obs_data["bcode_meta"]

    with open(args.parsimony_topology_file, "rb") as f:
        tree_topology_info = six.moves.cPickle.load(f)
        tree = tree_topology_info["tree"]

    for i in range(args.num_spr_moves):
        do_spr_move(tree, bcode_meta)

    with open(args.out_tree_file, "wb") as f:
        six.moves.cPickle.dump(tree_topology_info, f, protocol=2)

if __name__ == "__main__":
    main()
