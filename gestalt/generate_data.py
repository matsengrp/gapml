"""
A simulation engine to create cell lineage tree and data samples
"""
import sys
import numpy as np
import argparse
import time
import tensorflow as tf
import logging
import six
import pathlib
import os

from cell_state import CellState, CellTypeTree
from cell_lineage_tree import CellLineageTree
from cell_state_simulator import CellTypeSimulator
from clt_simulator import CLTSimulatorBifurcating
from clt_simulator_simple import CLTSimulatorSimple, CLTSimulatorSimpler, CLTSimulatorSimplest
from clt_likelihood_model import CLTLikelihoodModel
from allele_simulator_simult import AlleleSimulatorSimultaneous
from clt_observer import CLTObserver
from optim_settings import KnownModelParams
from barcode_metadata import BarcodeMetadata

from common import create_directory
from constants import NUM_BARCODE_V7_TARGETS, BARCODE_V7


def parse_args():
    parser = argparse.ArgumentParser(description='simulate GESTALT')
    parser.add_argument(
        '--out-obs-file',
        type=str,
        default="_output/obs_data.pkl",
        help='pkl file name for output file with observations')
    parser.add_argument(
        '--out-model-file',
        type=str,
        default="_output/true_model.pkl",
        help='pkl file name for output file with true_model')
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/generate_log.txt",
        help='name of log file')
    parser.add_argument(
        '--num-barcodes',
        type=int,
        default=1,
        help="number of independent barcodes. we assume all the same")
    parser.add_argument(
        '--target-lambdas',
        type=str,
        default=",".join([".4", ".5", ".1", "0.5", "0.3", "0.6"]),
        help='target cut rates -- will get slightly perturbed for the true value')
    parser.add_argument(
        '--target-lam-decay-rate',
        type=float,
        default=0,
        help='param for how much the hazard rate decays (right now is a linear decay)')
    parser.add_argument(
        '--perturb-target-lambdas-variance',
        type=float,
        default=0.002,
        help='variance of perturbations to target lambdas (so variance of perturbations)')
    parser.add_argument(
        '--double-cut-weight',
        type=float,
        default=2,
        help='Weight for double cuts')
    parser.add_argument(
        '--boost-probs',
        type=float,
        nargs=3,
        default=[0.1, 0.45, 0.45],
        help="""
        probability of boosting the insertion, left deletion, right deletion lengths.
        This boost is mutually exclusive -- the boost can be only applied to one of these categories.
        The boost is of length one right now.
        The boost is to ensure the target lambdas indicate the hazard of introducing a positive indel,
        not a no-op indel.
        """)
    parser.add_argument(
        '--trim-long-factor',
        type=float,
        nargs=2,
        default=[0.05] * 2,
        help='probability of doing no deletion/insertion during repair')
    parser.add_argument(
        '--trim-zero-probs',
        type=float,
        nargs=2,
        default=[0.01] * 2,
        help='probability of doing no deletion/insertion during repair')
    parser.add_argument(
        '--trim-poissons',
        type=float,
        nargs=2,
        default=[4] * 2,
        help='poisson parameter for distribution of symmetric deltion about cut site(s)'
    )
    parser.add_argument(
        '--insert-zero-prob',
        type=float,
        default=0.01,
        help='probability of doing no deletion/insertion during repair')
    parser.add_argument(
        '--insert-poisson',
        type=float,
        default=1,
        help='poisson parameter for distribution of insertion in cut site(s)')
    parser.add_argument(
        '--birth-sync-rounds', type=float, default=2, help='number of syncronous birth rounds before async begins')
    parser.add_argument(
        '--birth-sync-time', type=float, default=0.1, help='amount of time between syncronous birth rounds')
    parser.add_argument(
        '--birth-decay', type=float, default=-0.3, help='birth rate decay')
    parser.add_argument(
        '--birth-min', type=float, default=2, help='birth rate minimum')
    parser.add_argument(
        '--death-lambda', type=float, default=0.001, help='death rate')
    parser.add_argument(
        '--time', type=float, default=1, help='how much time to simulate')
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=1.0,
        help='proportion cells sampled/alleles successfully sequenced')
    parser.add_argument(
        '--debug', action='store_true', help='debug tensorflow')
    parser.add_argument(
        '--model-seed',
        type=int,
        default=0,
        help="Seed for generating the model")
    parser.add_argument(
        '--data-seed',
        type=int,
        default=0,
        help="Seed for generating data")
    parser.add_argument('--min-uniq-alleles', type=int, default=3)
    parser.add_argument('--max-uniq-alleles', type=int, default=10)
    parser.add_argument('--max-clt-nodes', type=int, default=1000000)
    parser.add_argument(
        '--max-abundance',
        type=int,
        default=None,
        help="Maximum abundance of the observed leaves")
    parser.add_argument(
        '--max-tries',
        type=int,
        default=1,
        help="Maximum tries to generate tree")
    parser.add_argument(
        '--is-one-leaf',
        action='store_true',
        help="special tree structure for tests")
    parser.add_argument(
        '--is-stupid-cherry',
        action='store_true',
        help="special tree structure for tests")

    parser.set_defaults()
    args = parser.parse_args()

    create_directory(args.out_obs_file)

    if args.is_one_leaf or args.is_stupid_cherry:
        assert args.sampling_rate == 1

    args.target_lambdas = [float(x) for x in args.target_lambdas.split(",")]
    args.num_targets = len(args.target_lambdas)

    assert args.time > args.birth_sync_rounds * args.birth_sync_time

    return args

def create_cell_type_tree(args):
    # This first rate means nothing!
    cell_type_tree = CellTypeTree(cell_type=0, rate=0.1)
    args.cell_rates = [0.20, 0.25, 0.15, 0.15]
    cell1 = CellTypeTree(cell_type=1, rate=args.cell_rates[0])
    cell2 = CellTypeTree(cell_type=2, rate=args.cell_rates[1])
    cell3 = CellTypeTree(cell_type=3, rate=args.cell_rates[2])
    cell4 = CellTypeTree(cell_type=4, rate=args.cell_rates[3])
    cell_type_tree.add_child(cell1)
    cell_type_tree.add_child(cell2)
    cell2.add_child(cell3)
    cell1.add_child(cell4)
    return cell_type_tree

def create_simulators(args, clt_model):
    allele_simulator = AlleleSimulatorSimultaneous(
            clt_model,
            np.array(args.boost_probs))
    cell_type_simulator = CellTypeSimulator(clt_model.cell_type_tree)
    if args.is_one_leaf:
        clt_simulator = CLTSimulatorSimplest(
                cell_type_simulator,
                allele_simulator)
    elif args.is_stupid_cherry:
        clt_simulator = CLTSimulatorSimpler(
                cell_type_simulator,
                allele_simulator)
    else:
        clt_simulator = CLTSimulatorBifurcating(
            args.birth_sync_rounds,
            args.birth_sync_time,
            args.birth_decay,
            args.birth_min,
            args.death_lambda,
            cell_type_simulator,
            allele_simulator,
            scale_hazard_func=lambda t: 1 - (t/args.time) * args.target_lam_decay_rate)
    observer = CLTObserver()
    return clt_simulator, observer

def create_cell_lineage_tree(
        args,
        clt_model: CLTLikelihoodModel,
        max_tries: int = 20,
        incr: float = 0.01,
        time_min: float = 1e-3):
    """
    @return original clt, the set of observed leaves, and the true topology for the observed leaves
    """
    clt_simulator, observer = create_simulators(args, clt_model)

    # Keep trying to make CLT until enough leaves in observed tree by modifying the max time of the tree
    birth_sync_time = args.birth_sync_time
    for i in range(max_tries):
        try:
            np.random.seed(args.model_seed)
            clt_simulator.birth_sync_time = birth_sync_time
            clt = clt_simulator.simulate_full_skeleton(
                tot_time=args.time,
                max_nodes=args.max_clt_nodes)
            clt.label_node_ids()
            # Now sample the leaves and create the true topology
            true_subtree = CLTObserver.sample_leaves(clt, args.sampling_rate)
            print("le....", len(true_subtree), birth_sync_time)
            if len(true_subtree) < args.min_uniq_alleles:
                birth_sync_time -= incr * np.random.rand()
            elif len(true_subtree) >= args.max_uniq_alleles:
                birth_sync_time += incr * np.random.rand()
            else:
                # We got a good number of leaves! Stop trying
                print("done!")
                break
        except ValueError as e:
            logging.info("ValueError warning.... %s", str(e))
            print("ValueError warning.... %s" % str(e))
            birth_sync_time += incr * np.random.rand()
            continue

    if len(true_subtree) < args.min_uniq_alleles:
        raise Exception("Could not manage to get enough leaves")

    np.random.seed(args.data_seed)
    clt_simulator.simulate_alleles(true_subtree)
    clt_simulator.simulate_cell_states(true_subtree)
    obs_leaves, obs_idx_to_leaves = observer.observe_leaves(
            true_subtree,
            seed=args.data_seed)

    logging.info(
        "birth time %f, num uniq alleles %d, num sampled leaves %d, num tot_leaves %d",
        birth_sync_time,
        len(obs_leaves),
        len(true_subtree),
        len(clt))

    true_subtree.label_tree_with_strs()
    logging.info(true_subtree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    logging.info(true_subtree.get_ascii(attributes=["node_id"], show_internal=True))
    logging.info(true_subtree.get_ascii(attributes=["dist"], show_internal=True))

    return obs_leaves, true_subtree, obs_idx_to_leaves

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(seed=args.model_seed)

    # Create the barcode in the embryo cell. Use default sequence if ten targets. Otherwise create a new one.
    barcode_orig = BarcodeMetadata.create_fake_barcode_str(args.num_targets) if args.num_targets != NUM_BARCODE_V7_TARGETS else BARCODE_V7
    bcode_meta = BarcodeMetadata(unedited_barcode = barcode_orig, num_barcodes = args.num_barcodes)

    # initialize the target lambdas with some perturbation to ensure we don't have eigenvalues that are exactly equal
    #args.target_lambdas, args.birth_intercept, args.death_lambda = initialize_lambda_rates(args)
    logging.info("args.target_lambdas %s" % str(args.target_lambdas))
    print("args.target_lambdas %s" % str(args.target_lambdas))

    # Create a cell-type tree
    cell_type_tree = create_cell_type_tree(args)

    logging.info(str(args))

    sess = tf.InteractiveSession()
    # Create model
    known_params = KnownModelParams(
            target_lams = True,
            double_cut_weight = True)
    clt_model = CLTLikelihoodModel(
            None,
            bcode_meta,
            sess,
            target_lams = np.array(args.target_lambdas),
            target_lam_decay_rate = np.array([args.target_lam_decay_rate]),
            known_params = known_params,
            double_cut_weight = [args.double_cut_weight],
            trim_long_factor = np.array(args.trim_long_factor),
            trim_zero_probs = np.array(args.trim_zero_probs),
            trim_short_poissons = np.array(args.trim_poissons),
            trim_long_poissons = np.array(args.trim_poissons),
            insert_zero_prob = np.array([args.insert_zero_prob]),
            insert_poisson = np.array([args.insert_poisson]),
            cell_type_tree = cell_type_tree,
            tot_time = args.time,
            tot_time_extra = 1e-10)
    tf.global_variables_initializer().run()
    logging.info("Done creating model")

    # Generate data!
    obs_leaves, true_subtree, obs_idx_to_leaves = create_cell_lineage_tree(
            args,
            clt_model,
            args.max_tries)

    # Check that the the abundance of the leaves is not too high
    if args.max_abundance is not None:
        for obs in obs_leaves:
            print(obs.abundance) #, str(obs))
            if obs.abundance > args.max_abundance:
                raise ValueError(
                    "Number of barcodes does not seem to be enough. There are leaves with %d abundance" % obs.abundance)

    logging.info("Number of uniq obs alleles %d", len(obs_leaves))
    print("Number of uniq obs alleles %d" % len(obs_leaves))

    # Save the observed data
    with open(args.out_obs_file, "wb") as f:
        out_dict = {
            "bcode_meta": bcode_meta,
            "obs_leaves": obs_leaves,
            "time": args.time,
        }
        six.moves.cPickle.dump(out_dict, f, protocol = 2)

    # Save the true data
    with open(args.out_model_file, "wb") as f:
        out_dict = {
            "true_model_params": clt_model.get_vars_as_dict(),
            "true_subtree": true_subtree,
            "obs_idx_to_leaves": obs_idx_to_leaves,
            "bcode_meta": bcode_meta,
            "time": args.time,
            "args": args}
        six.moves.cPickle.dump(out_dict, f, protocol = 2)

if __name__ == "__main__":
    main()
