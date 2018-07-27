"""
Helper code for splitting trees into training and validation sets
This is our version of training validation split.
It's the usual method if we have more than one barcode.
Otherwise we subsample the tree.
"""
from typing import Set
import numpy as np

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
import collapsed_tree

def create_train_val_tree(tree: CellLineageTree, bcode_meta: BarcodeMetadata, train_split: float):
    """
    @param train_split: fraction of data to reserve for training set

    @return CLT for training data, CLT for validation data, train bcode metadata, val bcode metadata
    """
    if bcode_meta.num_barcodes == 1:
        train_tree, val_tree, train_bcode, val_bcode = _create_train_val_tree_by_subsampling(
                tree,
                bcode_meta,
                train_split)
    else:
        train_tree, val_tree, train_bcode, val_bcode = _create_train_val_tree_by_barcode(
                tree, bcode_meta, train_split)
    train_tree.label_node_ids()
    val_tree.label_node_ids()
    return train_tree, val_tree, train_bcode, val_bcode

"""
Code for train/val split for >1 barcode
"""
def _restrict_barcodes(clt: CellLineageTree, min_bcode_idx: int, bcode_meta: BarcodeMetadata):
    """
    @param min_bcode_idx: the start index of the barcode we observe
    @param bcode_meta: barcode metadata, indicates how many barcodes we should restrict it to

    Update the alleles for each node in the tree to correspond to only the barcodes indicated
    """
    for node in clt.traverse():
        if node.allele_list is not None:
            node.set_allele_list(node.allele_list[min_bcode_idx: min_bcode_idx + bcode_meta.num_barcodes])
        else:
            node.allele_events_list = node.allele_events_list[min_bcode_idx: min_bcode_idx + bcode_meta.num_barcodes]
    clt.label_tree_with_strs()
    return clt

def _create_train_val_tree_by_barcode(clt: CellLineageTree, bcode_meta: BarcodeMetadata, train_split_rate: float):
    """
    There are multiple barcodes, so just partition the barcodes into train vs validation

    @return CLT for training data, CLT for validation data
    """
    num_train_bcodes = int(bcode_meta.num_barcodes * train_split_rate)
    num_val_bcodes = bcode_meta.num_barcodes - num_train_bcodes
    assert num_train_bcodes > 0 and num_val_bcodes > 0

    train_bcode_meta = BarcodeMetadata(
            bcode_meta.unedited_barcode,
            num_train_bcodes,
            bcode_meta.cut_site,
            bcode_meta.crucial_pos_len)
    train_clt = _restrict_barcodes(clt.copy(), 0, train_bcode_meta)
    val_bcode_meta = BarcodeMetadata(
            bcode_meta.unedited_barcode,
            num_val_bcodes,
            bcode_meta.cut_site,
            bcode_meta.crucial_pos_len)
    val_clt = _restrict_barcodes(clt.copy(), num_train_bcodes, val_bcode_meta)
    return train_clt, val_clt, train_bcode_meta, val_bcode_meta


"""
Code for train/val split for only 1 barcode
"""
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

def _create_train_val_tree_by_subsampling(clt: CellLineageTree, bcode_meta: BarcodeMetadata, train_split_rate: float):
    """
    Take a tree and randomly partition leaves into training vs validation
    Use the corresponding subtrees as a train CLT and a val CLT

    @return CLT for training data, CLT for validation data
    """
    # Assign leaf ids to train vs validation
    train_leaf_ids = set()
    val_leaf_ids = set()
    for leaf in clt:
        is_train = np.random.binomial(1, train_split_rate)
        if is_train:
            train_leaf_ids.add(leaf.node_id)
        else:
            val_leaf_ids.add(leaf.node_id)

    train_clt = _prune_tree(clt.copy(), train_leaf_ids)
    val_clt = _prune_tree(clt.copy(), val_leaf_ids)
    return train_clt, val_clt, bcode_meta, bcode_meta
