"""
Get newick from tune_topology.py
"""
import sys
import six
import os
import argparse
import json


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='get newick file')

    parser.add_argument(
        '--fitted-model-file',
        default='_output/tune_topology_fitted.pkl',
        help="The model to read in")
    parser.add_argument(
        '--out-newick-format',
        type=int,
        default=1,
        help="Output newick format, Select number from here http://etetoolkit.org/docs/latest/tutorial/tutorial_trees.html")
    parser.add_argument(
        '--out-newick-file',
        type=str,
        default="_output/tuned_tree.nw",
        help="File to output with tree info")
    parser.add_argument(
        '--out-leaf-file',
        type=str,
        default="_output/leaf.json",
        help="File to output with leaf info")

    parser.set_defaults(tot_time_known=True)
    args = parser.parse_args(args)
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)

    with open(args.fitted_model_file, "rb") as f:
        res_dict = six.moves.cPickle.load(f)
        fitted_res = res_dict["final_fit"]

    leaf_dict = {}
    for leaf_id, leaf in enumerate(fitted_res.fitted_bifurc_tree):
        leaf.name = "Leaf_%d" % leaf_id
        leaf_dict[leaf.name] = {
            "barcode_%d" % bcode_id:
                [{
                    "start_pos": evt.start_pos,
                    "del_len": evt.del_len,
                    "min_target": evt.min_target,
                    "max_target": evt.max_target,
                    "insert_str": evt.insert_str,
                    } for evt in allele_evts.events]
            for bcode_id, allele_evts in enumerate(leaf.allele_events_list)
            }

    fitted_res.fitted_bifurc_tree.write(format=args.out_newick_format, outfile=args.out_newick_file)
    with open(args.out_leaf_file, "w") as f:
        json.dump(leaf_dict, f, indent=1)

if __name__ == "__main__":
    main()
