"""
Apply filtration -- we can only observe the first `n` alleles in each cell.
"""
import sys
import argparse
import logging
import six
from typing import List

from clt_observer import ObservedAlignedSeq
from common import create_directory, save_data


def parse_args():
    parser = argparse.ArgumentParser(description='Collapse data based on first n alleles')
    parser.add_argument(
        '--obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file with observed sequence data, should be a dict with ObservedAlignSeq')
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/restrict_log.txt",
        help='pkl file with true model')
    parser.add_argument(
        '--num-barcodes',
        type=int,
        default=1,
        help="Number of the barcodes we actually observe")
    parser.add_argument(
        '--observe-cell-state',
        action='store_true',
        help='Do we observe cell state?')
    parser.add_argument(
        '--min-leaves',
        type=int,
        default=1,
        help="""
        Minimum number of leaves we want after restricting. Will consider incrementing to a later barcode
        if minimum is not achieved
        """)
    parser.add_argument(
        '--out-obs-file',
        type=str,
        default="_output/obs_data_b1.pkl",
        help='name of the output pkl file with collapsed observations')

    args = parser.parse_args()
    create_directory(args.out_obs_file)
    return args


def collapse_obs_leaves_by_first_alleles(
        obs_leaves: List[ObservedAlignedSeq],
        num_barcodes: int,
        min_leaves: int,
        observe_cell_state: bool):
    """
    Collapse the observed data based on the first `num_barcodes` alleles
    @return List[ObservedAlignedSeq]
    """
    min_barcode = 0
    num_obs = 0
    while num_obs < min_leaves:
        obs_dict = {}
        for obs in obs_leaves:
            if obs.allele_list is not None:
                obs.set_allele_list(obs.allele_list.create_truncated_version(
                    num_barcodes,
                    min_barcode=min_barcode))
            else:
                obs.allele_events_list = obs.allele_events_list[min_barcode: min_barcode + num_barcodes]

            # Make sure to keep unique observations and update abundance accordingly
            obs_key = str(obs) if observe_cell_state else obs.get_allele_str()
            if obs_key in obs_dict:
                obs_dict[obs_key].abundance += obs.abundance
            else:
                obs_dict[obs_key] = obs

        # Check number of observations
        num_obs = len(obs_dict.values())
        logging.info("Min barcode idx %d results in %d observations", min_barcode, num_obs)
        min_barcode += 1

    return list(obs_dict.values())


def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(str(args))

    with open(args.obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    bcode_meta = obs_data_dict["bcode_meta"]
    orig_num_barcodes = bcode_meta.num_barcodes
    assert args.num_barcodes <= orig_num_barcodes

    # Update the barcode metadata to have the collapsed number of barcodes
    bcode_meta.num_barcodes = args.num_barcodes

    # Now start collapsing the observations by first n alleles
    raw_obs_leaves = obs_data_dict["obs_leaves"]
    obs_data_dict["obs_leaves"] = collapse_obs_leaves_by_first_alleles(
            raw_obs_leaves,
            args.num_barcodes,
            args.min_leaves,
            args.observe_cell_state)
    print(
        "Number of uniq obs after restricting to first %d alleles: %d" %
        (args.num_barcodes,
        len(obs_data_dict["obs_leaves"])))
    logging.info(
        "Number of uniq obs after restricting to first %d alleles: %d",
        args.num_barcodes,
        len(obs_data_dict["obs_leaves"]))
    save_data(obs_data_dict, args.out_obs_file)


if __name__ == "__main__":
    main()
