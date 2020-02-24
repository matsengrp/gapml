import six
from typing import List

from model_assessor import ModelAssessor


def read_data(
        obs_file: str,
        topology_file: str = None,
        leaf_key: str= "leaf_key"):
    """
    Read the data files...
    @param new_leaf_key: use this new leaf attribute as the unique leaf identifier
    """
    with open(obs_file, "rb") as f:
        obs_data_dict = six.moves.cPickle.load(f)

    if topology_file is not None:
        with open(topology_file, "rb") as f:
            tree_topology_info = six.moves.cPickle.load(f)
            tree = tree_topology_info["tree"]
            tree.label_node_ids()

        # Mark the leaves with a unique id in case we need to compare against the true model
        for leaf in tree:
            leaf.add_feature(leaf_key, leaf.allele_events_list_str)

        # If this tree is not unresolved, then mark all the multifurcations as resolved
        if not tree_topology_info["multifurc"]:
            for node in tree.traverse():
                node.resolved_multifurcation = True
    else:
        tree = None

    return tree, obs_data_dict


def read_true_model(
        true_model_file: str,
        n_bcodes: int,
        measurer_classes: List = [],
        scratch_dir: str = None,
        use_error_prone_alleles: bool = False,
        leaf_key: str = "leaf_key"):
    """
    @param n_bcodes: the number of barcodes to restrict to when loading the true model

    If true model files available, read them
    """
    with open(true_model_file, "rb") as f:
        true_model = six.moves.cPickle.load(f)

    assessor = None
    if len(measurer_classes):
        true_tree = true_model["true_subtree"]

        # If we are using an alternate reference leaf key, copy over the relevant quantities
        if use_error_prone_alleles:
            for leaf in true_tree:
                leaf.allele_events_list = leaf.allele_events_list
                leaf.allele_list = leaf.allele_list

        # Restrict the number of observed barcodes
        true_tree.restrict_barcodes(range(n_bcodes))

        # Mark the leaves with a unique id for when we compare the fitted
        # against the true tree
        for leaf in true_tree:
            leaf.add_feature(leaf_key, leaf.allele_events_list_str)

        assessor = ModelAssessor(
            true_model["true_model_params"],
            true_tree,
            measurer_classes,
            scratch_dir,
            leaf_key=leaf_key)
    return true_model["true_model_params"], assessor
