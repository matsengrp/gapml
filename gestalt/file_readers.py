import six
from typing import List

from tree_distance import TreeDistanceMeasurerAgg


def read_data(obs_file: str, topology_file: str):
    """
    Read the data files...
    """
    with open(obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    with open(topology_file, "rb") as f:
        tree_topology_info = six.moves.cPickle.load(f)
        tree = tree_topology_info["tree"]
        tree.label_node_ids()

    # If this tree is not unresolved, then mark all the multifurcations as resolved
    if not tree_topology_info["multifurc"]:
        for node in tree.traverse():
            node.resolved_multifurcation = True

    return tree, obs_data_dict


def read_true_model(
        true_model_file: str,
        n_bcodes: int,
        measurer_classes: List = [],
        scratch_dir: str = None):
    """
    If true model files available, read them
    """
    # TODO: take the tree loading comparison code out of plot code
    with open(true_model_file, "rb") as f:
        true_model = six.moves.cPickle.load(f)

    oracle_dist_measurers = None
    if len(measurer_classes):
        oracle_dist_measurers = TreeDistanceMeasurerAgg.create_single_abundance_measurer(
            true_model["true_subtree"],
            n_bcodes,
            measurer_classes,
            scratch_dir)
    return true_model["true_model_params"], oracle_dist_measurers
