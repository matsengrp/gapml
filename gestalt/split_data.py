"""
Helper code for splitting trees into training and validation sets
This is our version of k-fold CV for trees
"""
from typing import Dict, List
import numpy as np
import logging
import random

from sklearn.model_selection import KFold

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata


class TreeDataSplit:
    """
    Stores the metadata and data for each "fold" from our variant of k-fold CV
    """
    def __init__(
            self,
            train_clt: CellLineageTree,
            train_bcode_meta: BarcodeMetadata,
            val_clt: CellLineageTree,
            val_bcode_meta: BarcodeMetadata):
        """
        @param is_kfold_tree: True = we split the tree into subtrees (for the 1 bcode case)
        """
        self.train_clt = train_clt
        self.train_bcode_meta = train_bcode_meta
        self.val_clt = val_clt
        self.val_bcode_meta = val_bcode_meta


def _pick_random_validation_node_ids(tree: CellLineageTree, min_multifurc_children: int):
    val_obs = []
    for node in tree.traverse():
        if node.is_leaf():
            continue

        if len(node.get_children()) <= min_multifurc_children:
            # Only create validation leaves from multifurcations with
            # 4 or more children
            continue

        leaf_children = [c for c in node.get_children() if c.is_leaf()]
        leaf = random.choice(leaf_children)
        val_obs.append((leaf.copy(), leaf.up.node_id))
    return val_obs


def create_kfold_trees(
        tree: CellLineageTree,
        bcode_meta: BarcodeMetadata,
        n_splits: int,
        min_multifurc_children: int = 3):
    """
    Take a tree and create k-fold datasets based on children of the root node
    This takes a tree and creates subtrees by taking a subset of the leaves,
    We refer to this as a "fold" of the tree

    @param n_splits: the number of folds requested

    @return List[TreeDataSplit] corresponding to each fold
    """
    all_train_trees = []
    for i in range(n_splits):
        tree_copy = tree.copy()

        val_obs = _pick_random_validation_node_ids(tree_copy, min_multifurc_children)
        val_obs_ids = set([n.node_id for n, _ in val_obs])
        if len(val_obs) == 0:
            # If cannot find enough validation leaves, try a smaller threshold
            val_obs = _pick_random_validation_node_ids(tree_copy, max(min_multifurc_children - 1, 3))
            val_obs_ids = set([n.node_id for n, _ in val_obs])
        assert len(val_obs) > 0

        # Prune tree to create our tree for training
        train_leaf_ids = set()
        for leaf in tree_copy:
            if leaf.node_id not in val_obs_ids:
                train_leaf_ids.add(leaf.node_id)
        train_tree = CellLineageTree.prune_tree(tree_copy, train_leaf_ids)

        for node in train_tree.traverse():
            node.add_feature("orig_node_id", node.node_id)
        num_train_tree_nodes = train_tree.label_node_ids()

        # Create the validation tree -- just attach back on our validation leaves
        val_tree = train_tree.copy()
        for leaf_idx, (orig_leaf, leaf_par_id) in enumerate(val_obs):
            leaf = orig_leaf.copy()
            leaf.add_feature("node_id", leaf_idx + num_train_tree_nodes)
            parent_node = val_tree.search_nodes(orig_node_id=leaf_par_id)[0]
            parent_node.add_child(leaf)

        # Clean up the extra attribute
        for node in train_tree.traverse():
            node.del_feature("orig_node_id")

        all_train_trees.append(TreeDataSplit(
            train_tree,
            bcode_meta,
            val_tree,
            bcode_meta))

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
    for bcode_idxs, val_idxs in kf.split(np.arange(bcode_meta.num_barcodes)):
        logging.info("Train fold barcode idxs %s", bcode_idxs)
        num_train_bcodes = len(bcode_idxs)
        train_bcode_meta = BarcodeMetadata(
                bcode_meta.unedited_barcode,
                num_train_bcodes,
                bcode_meta.cut_site,
                bcode_meta.crucial_pos_len)
        val_bcode_meta = BarcodeMetadata(
                bcode_meta.unedited_barcode,
                len(val_idxs),
                bcode_meta.cut_site,
                bcode_meta.crucial_pos_len)
        train_clt = tree.copy()
        train_clt.restrict_barcodes(bcode_idxs)
        val_clt = tree.copy()
        val_clt.restrict_barcodes(val_idxs)
        all_train_trees.append(TreeDataSplit(
            train_clt,
            train_bcode_meta,
            val_clt=val_clt,
            val_bcode_meta=val_bcode_meta))

    return all_train_trees
