"""
Helper code for splitting trees into training and validation sets
This is our version of k-fold CV for trees
"""
from typing import Dict, List
import numpy as np
from numpy import ndarray
import logging
import random

from sklearn.model_selection import KFold

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata


class TreeDataSplit:
    """
    Stores the metadata and data for each "fold" from our variant of k-fold CV
    """
    def __init__(self,
            tree: CellLineageTree,
            val_obs: List[CellLineageTree],
            bcode_meta: BarcodeMetadata,
            node_to_orig_id: Dict[int, int],
            is_kfold_tree: bool = False):
        """
        @param is_kfold_tree: True = we split the tree into subtrees (for the 1 bcode case)
        """
        self.tree = tree
        self.val_obs = val_obs
        self.bcode_meta = bcode_meta
        self.node_to_orig_id = node_to_orig_id
        self.is_kfold_tree = is_kfold_tree

def create_kfold_trees(tree: CellLineageTree, bcode_meta: BarcodeMetadata, n_splits: int):
    """
    Take a tree and create k-fold datasets based on children of the root node
    This takes a tree and creates subtrees by taking a subset of the leaves,
    We refer to this as a "fold" of the tree

    @param n_splits: the number of folds requested

    @return List[TreeDataSplit] corresponding to each fold
    """
    all_train_trees = []
    for i in range(n_splits):
        print("split")
        tree_copy = tree.copy()

        val_obs = []
        for node in tree_copy.traverse():
            if node.is_leaf():
                continue

            if len(node.get_children()) <= 3:
                continue

            leaf_children = [c for c in node.get_children() if c.is_leaf()]
            leaf = random.choice(leaf_children)
            val_obs.append((
                leaf.copy(),
                leaf.up.node_id))
            logging.info('val obs %d', leaf.node_id)
            print("leaf id", leaf.node_id, "parent node id", leaf.up.node_id)

        val_obs_ids = set([n.node_id for n, _ in val_obs])
        train_leaf_ids = set()
        for leaf in tree_copy:
            if leaf.node_id not in val_obs_ids:
                train_leaf_ids.add(leaf.node_id)

        train_tree = CellLineageTree.prune_tree(tree_copy, train_leaf_ids)
        logging.info("before")
        logging.info("SAMPLED TREE")
        logging.info(train_tree.get_ascii(attributes=["node_id"], show_internal=True))

        for node in train_tree.traverse():
            node.add_feature("orig_node_id", node.node_id)
        train_tree.label_node_ids()

        node_to_orig_id = dict()
        for node in train_tree.traverse():
            node_to_orig_id[node.node_id] = node.orig_node_id

        all_train_trees.append(TreeDataSplit(
            train_tree,
            val_obs,
            bcode_meta,
            node_to_orig_id,
            is_kfold_tree=True))

    return all_train_trees


def create_kfold_barcode_trees(tree: CellLineageTree, bcode_meta: BarcodeMetadata, n_splits: int):
    """
    Take a tree and create k-fold datasets by splitting on the independent barcodes.
    We refer to this as a "barcode-fold" of the tree

    @param n_splits: the number of folds requested

    @return List[TreeDataSplit] corresponding to each fold
    """
    raise NotImplementedError("not yet. doesnt calc Pr(V|T)")
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
            train_bcode_meta,
            is_kfold_tree=False))

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
