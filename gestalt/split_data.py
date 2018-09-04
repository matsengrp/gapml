"""
Helper code for splitting trees into training and validation sets
This is our version of k-fold CV for trees
"""
from typing import Set, Dict
import numpy as np
from numpy import ndarray
import logging

from sklearn.model_selection import KFold

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
import collapsed_tree

class TreeDataSplit:
    """
    Stores the metadata and data for each "fold" from our variant of k-fold CV
    """
    def __init__(self,
            tree: CellLineageTree,
            bcode_meta: BarcodeMetadata):
        self.tree = tree
        self.bcode_meta = bcode_meta

def create_kfold_trees(tree: CellLineageTree, bcode_meta: BarcodeMetadata, n_splits: int):
    """
    Take a tree and create k-fold datasets based on children of the root node
    This takes a tree and creates subtrees by taking a subset of the leaves,
    We refer to this as a "fold" of the tree

    @param n_splits: the number of folds requested

    @return List[TreeDataSplit] corresponding to each fold
    """
    assert len(tree.get_children()) > 1 and n_splits >= 2

    # Assign by splitting on children of root node -- perform k-fold cv
    # This decreases the correlation between training sets
    children = tree.get_children()
    children_indices = [c.node_id for c in children]
    n_splits = min(int(len(children_indices)/2), n_splits)
    logging.info("Splitting tree into %d, total %d children", n_splits, len(children))
    assert n_splits > 1

    kf = KFold(n_splits=n_splits, shuffle=True)
    all_train_trees = []
    for fold_indices, _ in kf.split(children_indices):
        # Now actually assign the leaf nodes appropriately
        train_leaf_ids = set()
        for child_idx in fold_indices:
            train_leaf_ids.update([l.node_id for l in children[child_idx]])

        train_tree = _prune_tree(tree.copy(), train_leaf_ids)
        logging.info("SAMPLED TREE")
        logging.info(train_tree.get_ascii(attributes=["node_id"], show_internal=True))

        for leaf in train_tree:
            leaf.add_feature("orig_node_id", leaf.node_id)
        train_tree.label_node_ids()

        node_to_orig_id = dict()
        for leaf in train_tree:
            node_to_orig_id[leaf.node_id] = leaf.orig_node_id

        all_train_trees.append(TreeDataSplit(
            train_tree,
            bcode_meta,
            node_to_orig_id))

    return all_train_trees

def create_kfold_barcode_trees(tree: CellLineageTree, bcode_meta: BarcodeMetadata, n_splits: int):
    """
    Take a tree and create k-fold datasets by splitting on the independent barcodes.
    We refer to this as a "barcode-fold" of the tree

    @param n_splits: the number of folds requested

    @return List[TreeDataSplit] corresponding to each fold
    """
    # Assign by splitting on children of root node -- perform k-fold cv
    # This decreases the correlation between training sets
    logging.info("Splitting barcode into %d splits", n_splits)
    assert n_splits > 1

    kf = KFold(n_splits=n_splits, shuffle=True)
    all_train_trees = []
    for bcode_idxs, _ in kf.split(np.arange(bcode_meta.num_barcodes)):
        logging.info("Train fold barcode idxs %s", bcode_idxs)
        num_train_bcodes = len(bcode_idxs)
        train_bcode_meta = BarcodeMetadata(
                bcode_meta.unedited_barcode,
                num_train_bcodes,
                bcode_meta.cut_site,
                bcode_meta.crucial_pos_len)
        train_clt = _restrict_barcodes(tree.copy(), bcode_idxs)
        all_train_trees.append(TreeDataSplit(
            train_clt,
            train_bcode_meta))

    return all_train_trees

def _restrict_barcodes(clt: CellLineageTree, bcode_idxs: ndarray):
    """
    @param bcode_idxs: the indices of the barcodes we observe
    Update the alleles for each node in the tree to correspond to only the barcodes indicated
    """
    for node in clt.traverse():
        node.allele_events_list = [node.allele_events_list[i] for i in bcode_idxs]
    clt.label_tree_with_strs()
    return clt

def _prune_tree(clt: CellLineageTree, keep_leaf_ids: Set[int]):
    """
    prune the tree to only keep the leaves indicated
    custom pruning (cause ete was doing weird things...)
    """
    for node in clt.iter_descendants():
        if sum((node2.node_id in keep_leaf_ids) for node2 in node.traverse()) == 0:
            node.detach()
    collapsed_tree._remove_single_child_unobs_nodes(clt)
    return clt
