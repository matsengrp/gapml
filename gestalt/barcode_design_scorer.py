"""
Tunes penalty params, tree topology, and model params
"""
import sys
import six
import os
import argparse
import logging
import numpy as np

from cell_lineage_tree import CellLineageTree
from tree_distance import BHVDistanceMeasurer, InternalCorrMeasurer
from model_assessor import ModelAssessor


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='make barcode scores')
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log_tune_topology.txt")
    parser.add_argument(
        '--true-model-file',
        type=str,
        default="_output/true_model.pkl",
        help='pkl file with true model if available')
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")
    parser.set_defaults()
    args = parser.parse_args(args)

    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    with open(args.true_model_file, "rb") as f:
        true_model = six.moves.cPickle.load(f)

    true_subtree = true_model["true_subtree"]
    for idx, leaf in enumerate(true_subtree):
        leaf.allele_events_list_str = str(idx)

    assessor = ModelAssessor(
        true_model["true_model_params"],
        true_subtree,
        n_bcodes=None,
        tree_measurer_classes=[BHVDistanceMeasurer, InternalCorrMeasurer],
        scratch_dir=args.scratch_dir)
    true_subtree = assessor.ref_tree

    num_bins = 20
    tot_time = 1.
    node_time_dict = [{
        "nodes": [],
        "check_set": set()
    } for _ in range(num_bins + 1)]
    true_subtree.add_feature("dist_to_root", 0)
    true_subtree.label_node_ids()
    for node in true_subtree.get_descendants("preorder"):
        node.dist_to_root = node.up.dist_to_root + node.dist
        bin_num = int(np.floor(node.dist_to_root/tot_time * num_bins))
        if node.node_id in node_time_dict[bin_num]["check_set"]:
            continue
        else:
            node_time_dict[bin_num]["nodes"].append(node)
            node_time_dict[bin_num]["check_set"].add(node.node_id)
            node_time_dict[bin_num]["check_set"].update([
                descendant.node_id for descendant in node.get_descendants()])

    for idx, bin_dict in enumerate(node_time_dict):
        bin = bin_dict["nodes"]
        if len(bin) == 0:
            continue

        root_node = CellLineageTree(
            true_subtree.allele_list,
            true_subtree.allele_events_list,
            true_subtree.cell_state,
            dist=0)
        prev_node = root_node
        root_node.add_feature("dist_to_root", 0)
        sorted_nodes = sorted(bin, key=lambda node: node.dist_to_root)
        for node in sorted_nodes:
            spine_node = CellLineageTree(
                node.allele_list,
                node.allele_events_list,
                node.cell_state,
                dist=node.dist_to_root - prev_node.dist_to_root,
            )
            spine_node.add_features(
                dist_to_root=node.dist_to_root,
                allele_events_list_str=node.allele_events_list_str)
            prev_node.add_child(spine_node)
            for leaf in node:
                new_leaf = CellLineageTree(
                    leaf.allele_list,
                    leaf.allele_events_list,
                    leaf.cell_state,
                    dist=tot_time - node.dist_to_root,
                )
                new_leaf.add_feature("allele_events_list_str", leaf.allele_events_list_str)
                spine_node.add_child(new_leaf)
            prev_node = spine_node

        # print("num leaves", len(root_node))
        # print(root_node.get_ascii(attributes=["dist"]))
        tree_dist = assessor.assess(None, root_node)
        print("bin", idx, tree_dist)

    logging.info("Complete!!!")


if __name__ == "__main__":
    main()
