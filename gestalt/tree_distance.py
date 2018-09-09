import os
import subprocess
import numpy as np
from typing import List
import random
import time
import logging

from scipy.stats import spearmanr, kendalltau, pearsonr
from cell_lineage_tree import CellLineageTree
from collapsed_tree import _remove_single_child_unobs_nodes
from constant_paths import RSPR_PATH, BHV_PATH, TAU_GEO_JAR_PATH
import collapsed_tree


class TreeDistanceMeasurerAgg:
    """
    Aggregates tree distances
    """
    def __init__(
            self,
            ref_tree: CellLineageTree,
            measurer_classes: List,
            scratch_dir: str,
            do_expand_abundance: bool = False):
        """
        @param measurer_classes: list of classes (subclasses of TreeDistanceMeasurer)
                                to instantiate for tree distance measurements
                                ex. [SPRDistanceMeasurer]
        @param ref_tree: the reference tree to measure distances from
        @param scratch_dir: a scratch directory used by TreeDistanceMeasurer
        @param do_expand_abundance: for all tree comparisons, create leaves with abundance one
                                for each multi-abundance leaf
        """
        self.ref_tree = ref_tree
        self.measurers = [meas_cls(ref_tree, scratch_dir) for meas_cls in measurer_classes]
        self.do_expand_abundance = do_expand_abundance

    def get_tree_dists(self, trees: List[CellLineageTree]):
        """
        @return a list of dictionaries of tree distances for each tree to the ref_tree
        """
        all_dists = []
        for tree in trees:
            tree_dists = {}
            compare_tree = TreeDistanceMeasurerAgg.create_single_abundance_tree(tree) if self.do_expand_abundance else tree
            for measurer in self.measurers:
                dist = measurer.get_dist(compare_tree)
                tree_dists[measurer.name] = dist
            all_dists.append(tree_dists)
        return all_dists

    @staticmethod
    def create_single_abundance_measurer(
            raw_tree: CellLineageTree,
            measurer_classes: List,
            scratch_dir: str):
        """
        @param raw_tree: this tree may have more than the requested number of barcodes
                        to restrict to
        @param n_bcodes: number of barcodes to restrict in the true tree
        @return TreeDistanceMeasurer
        """
        ref_tree = raw_tree.copy()

        # First restrict down to the requested number of barcodes
        # Rename nodes if they have the same first few barcodes
        # TODO: better way to ID the tree leaves?
        existing_strs = {}
        for node in ref_tree:
            if node.allele_events_list_str in existing_strs:
                count = existing_strs[node.allele_events_list_str]
                existing_strs[node.allele_events_list_str] += 1
                node.allele_events_list_str = "%s==%d" % (
                         node.allele_events_list_str,
                         count)
            else:
                existing_strs[node.allele_events_list_str] = 1

        return TreeDistanceMeasurerAgg(
            ref_tree,
            measurer_classes,
            scratch_dir,
            do_expand_abundance=True)

    @staticmethod
    def create_single_abundance_tree(tree: CellLineageTree):
        """
        Create tree with appropriate number of leaves to match abundance
        Just attach with zero distance to the existing leaf node
        Now we can compare it to the tree created in the `create` func
        """
        leaved_tree = tree.copy()
        for node in leaved_tree:
            curr_node = node
            for idx in range(node.abundance - 1):
                new_child = CellLineageTree(
                    curr_node.allele_list,
                    curr_node.allele_events_list,
                    curr_node.cell_state,
                    dist=0,
                    abundance=1,
                    resolved_multifurcation=True)
                new_child.allele_events_list_str = "%s==%d" % (
                    new_child.allele_events_list_str,
                    idx + 1)
                copy_leaf = CellLineageTree(
                    curr_node.allele_list,
                    curr_node.allele_events_list,
                    curr_node.cell_state,
                    dist=0,
                    abundance=1,
                    resolved_multifurcation=True)
                copy_leaf.allele_events_list_str = curr_node.allele_events_list_str
                curr_node.add_child(new_child)
                curr_node.add_child(copy_leaf)
                curr_node = new_child
            node.abundance = 1
        return leaved_tree


class TreeDistanceMeasurer:
    """
    Class that measures distances btw trees -- subclass this!
    """
    def __init__(
            self,
            ref_tree: CellLineageTree,
            scratch_dir: str,
            attr: str = "allele_events_list_str"):
        """
        @param ref_tree: tree to measure dsitances from
        @param scratch_dir: a directory where files can be written to if needed
                            (not used by all subclasses)
        """
        self.ref_tree = ref_tree
        self.scratch_dir = scratch_dir
        self.attr = attr

    def _rename_nodes(self, tree):
        """
        Run rspr software from Chris Whidden
        Chris's software requires that the first tree be bifurcating.
        Second tree can be multifurcating.
        """
        # Set node name to the attribute
        _remove_single_child_unobs_nodes(self.ref_tree)
        if len(self.ref_tree.get_children()) == 1:
            child = self.ref_tree.get_children()[0]
            child.delete(prevent_nondicotomic=True, preserve_branch_length=True)
        for n in self.ref_tree.traverse():
            n.name = getattr(n, self.attr)

        _remove_single_child_unobs_nodes(tree)
        if len(tree.get_children()) == 1:
            child = tree.get_children()[0]
            child.delete(prevent_nondicotomic=True, preserve_branch_length=True)
        for n in tree.traverse():
            n.name = getattr(n, self.attr)

    def get_dist(self, tree: CellLineageTree):
        """
        @return single tree distance number
        """
        raise NotImplementedError("need to implement!")

    def group_trees_by_dist(self, trees: List[CellLineageTree], max_trees: int=None):
        """
        @return a dictionary mapping tree distance to a group of uniq trees
                with at most max_trees for each group
        """
        tree_group_dict = {}
        for tree in trees:
            dist = self.get_dist(tree)
            if dist in tree_group_dict:
                tree_group_dict[dist].append(tree)
            else:
                tree_group_dict[dist] = [tree]

        for dist, tree_group in tree_group_dict.items():
            tree_group_dict[dist] = self.get_uniq_trees(tree_group, max_trees=max_trees)

        return tree_group_dict

    @classmethod
    def get_uniq_trees(cls, trees: List[CellLineageTree], max_trees: int=None):
        """
        @param max_trees: find this many uniq trees at most
        @return tuple with:
                    uniq trees, at most max_trees
                    indices of these uniq trees from the original tree list
        """
        # Shuffle the trees in case we are taking the first couple trees only
        random.shuffle(trees)
        num_trees = 1
        uniq_trees = [trees[0]]
        if max_trees is not None and max_trees == 1:
            return uniq_trees

        tree_measurers = [cls(trees[0], self.scratch_dir)]
        for idx, tree in enumerate(trees[1:]):
            has_match = False
            for uniq_t, measurer in zip(uniq_trees, tree_measurers):
                dist = measurer.get_dist(tree)
                has_match = dist == 0
                if has_match:
                    break
            if not has_match:
                uniq_trees.append(tree)
                tree_measurers.append(cls(tree, self.scratch_dir))
                num_trees += 1
                if max_trees is not None and num_trees > max_trees:
                    break
        return uniq_trees


class UnrootRFDistanceMeasurer(TreeDistanceMeasurer):
    """
    Robinson foulds distance, unrooted trees
    """
    name = "ete_rf_unroot"

    def get_dist(self, tree):
        rf_res = self.ref_tree.robinson_foulds(
                tree,
                attr_t1=self.attr,
                attr_t2=self.attr,
                expand_polytomies=False,
                unrooted_trees=True)
        return rf_res[0]


class RootRFDistanceMeasurer(TreeDistanceMeasurer):
    """
    Robinson foulds distance, rooted trees
    """
    name = "ete_rf_root"

    def get_dist(self, tree):
        try:
            rf_res = self.ref_tree.robinson_foulds(
                    tree,
                    attr_t1=self.attr,
                    attr_t2=self.attr,
                    expand_polytomies=False,
                    unrooted_trees=False)
            return rf_res[0]
        except Exception as err:
            logging.info("cannot get root RF distance: %s", str(err))
            print("cannot get root RF distance: %s", str(err))
            return np.NaN


class BHVDistanceMeasurer(TreeDistanceMeasurer):
    """
    BHV distance
    """
    name = "bhv"

    def get_dist(self, raw_tree, attr="allele_events_list_str"):
        """
        http://comet.lehman.cuny.edu/owen/code.html
        """
        ref_tree = self.ref_tree.copy()
        if len(ref_tree.get_children()) == 1:
            ref_tree.get_children()[0].delete(prevent_nondicotomic=True, preserve_branch_length=True)
        for leaf in ref_tree:
            #leaf.dist = 0
            leaf.name = getattr(leaf, attr)

        tree = raw_tree.copy()
        if len(tree.get_children()) == 1:
            tree.get_children()[0].delete(prevent_nondicotomic=True, preserve_branch_length=True)
        for leaf in tree:
            #leaf.dist = 0
            leaf.name = getattr(leaf, attr)

        self._rename_nodes(tree)
        assert len(ref_tree) == len(tree)

        # Write tree out in newick format
        suffix = "%d%d" % (int(time.time()), np.random.randint(1000000))
        tree_in_file = "%s/tree_newick%s.txt" % (
                self.scratch_dir, suffix)
        with open(tree_in_file, "w") as f:
            f.write(ref_tree.write(format=5))
            f.write("\n")
            f.write(tree.write(format=5))
            f.write("\n")

        # Run bhv
        bhv_out_file = "%s/tree_bhv%s.txt" % (self.scratch_dir, suffix)
        verbose_out_file = "%s/tree_bhv%s_verbose.txt" % (self.scratch_dir, suffix)
        bhv_cmd = "java -jar %s -d -v -o %s %s > %s" % (
                BHV_PATH,
                bhv_out_file,
                tree_in_file,
                verbose_out_file)
        subprocess.check_output(
                bhv_cmd,
                shell=True)

        # read bhv output, distance is on the last line
        with open(bhv_out_file, "r") as f:
            lines = f.readlines()
            bhv_dist = float(lines[0].split("\t")[-1])

        os.remove(tree_in_file)
        os.remove(bhv_out_file)
        os.remove(verbose_out_file)
        return bhv_dist


class SPRDistanceMeasurer(TreeDistanceMeasurer):
    """
    SPR distance
    """
    name = "spr"

    def get_dist(self, tree):
        """
        Run rspr software from Chris Whidden
        Chris's software requires that the first tree be bifurcating.
        Second tree can be multifurcating.
        """
        for n in self.ref_tree.traverse():
            if not n.is_leaf() and len(n.get_children()) > 2:
                raise ValueError("Reference tree is not binary. SPR will not work")

        self._rename_nodes(tree)

        # Write tree out in newick format
        suffix = "%d%d" % (int(time.time()), np.random.randint(1000000))
        tree_in_file = "%s/tree_newick%s.txt" % (
                self.scratch_dir, suffix)
        with open(tree_in_file, "w") as f:
            f.write(self.ref_tree.write(format=9))
            f.write("\n")
            f.write(tree.write(format=9))
            f.write("\n")

        # Run rspr
        rspr_out_file = "%s/tree_spr%s.txt" % (self.scratch_dir, suffix)
        subprocess.check_output(
                "%s -fpt < %s > %s" % (RSPR_PATH, tree_in_file, rspr_out_file),
                shell=True)

        # Read rspr output, distance is on the last line
        with open(rspr_out_file, "r") as f:
            lines = f.readlines()
            spr_dist = int(lines[-1].split("=")[-1])

        os.remove(tree_in_file)
        os.remove(rspr_out_file)

        return spr_dist


class GavruskinMeasurer(TreeDistanceMeasurer):
    """
    "BHV" distance but for ultrametric trees
    The tau-metric trees
    https://github.com/gavruskin/tauGeodesic
    """
    name = "tau-bhv"

    def _perturb_distances(self, raw_tree, perturb_err=0.5 * 1e-4):
        tree = raw_tree.copy()
        for node in tree.traverse():
            node.dist += np.random.rand() * perturb_err
        return tree

    def get_dist(self, tree):
        """
        Uses a modification of the tauGeodesic code
        -- mostly modifies the code to deal with newick format instead
        """
        try_vals = [0, 1e-10, 1e-8, 1e-4 * 0.5, 1e-3]
        for perturb_err in try_vals:
            try:
                tree = self._perturb_distances(tree, perturb_err)
                self._rename_nodes(tree)

                # Write tree out in newick format
                suffix = "%d%d" % (int(time.time()), np.random.randint(1000000))

                ref_tree_in_file = "%s/ref_tree_newick%s.txt" % (
                        self.scratch_dir, suffix)
                with open(ref_tree_in_file, "w") as f:
                    f.write(self.ref_tree.write(format=5))

                tree_in_file = "%s/tree_newick%s.txt" % (
                        self.scratch_dir, suffix)
                with open(tree_in_file, "w") as f:
                    f.write(tree.write(format=5))

                # Run geodesic distance code
                out_file = "%s/tree_tau_ultra%s.txt" % (self.scratch_dir, suffix)
                tau_cmd = "java -jar %s %s %s > %s 2>&1" % (TAU_GEO_JAR_PATH, ref_tree_in_file, tree_in_file, out_file)
                subprocess.check_output(
                        tau_cmd,
                        shell=True)

                # read tau-distance output, distance is on the last line
                with open(out_file, "r") as f:
                    lines = f.readlines()
                    tau_dist = float(lines[-1])
                print("good err", perturb_err)
                return tau_dist
            except subprocess.CalledProcessError as e:
                print(e)
            finally:
                os.remove(ref_tree_in_file)
                os.remove(tree_in_file)
                os.remove(out_file)
        raise ValueError("Failed to calculate tau dist")


class MRCADistanceMeasurer(TreeDistanceMeasurer):
    """
    Use mrca dist in "Mapping Phylogenetic Trees to Reveal Distinct Patterns of Evolution", Kendall and Colijn
    """
    name = "mrca"

    def __init__(
            self,
            ref_tree: CellLineageTree,
            scratch_dir: str = None,
            attr="allele_events_list_str"):
        self.ref_tree = ref_tree
        self.scratch_dir = scratch_dir
        self.attr = attr

        # Number the leaves because we need to represent each tree by its pairwise
        # MRCA distance matrix
        leaf_str_sorted = [getattr(leaf, attr) for leaf in ref_tree]
        self.leaf_dict = {}
        for idx, leaf_str in enumerate(leaf_str_sorted):
            self.leaf_dict[leaf_str] = idx
        self.num_leaves = len(ref_tree)

        self.ref_tree_mrca_matrix = self._get_mrca_matrix(ref_tree)

    def _get_mrca_matrix(self, tree, perturb=0):
        """
        @return the pairwise MRCA distance matrix for that tree
        """
        mrca_matrix = np.zeros((self.num_leaves, self.num_leaves))
        assert len(tree) == self.num_leaves
        for leaf1 in tree:
            for leaf2 in tree:
                leaf1_idx = self.leaf_dict[getattr(leaf1, self.attr)]
                leaf2_idx = self.leaf_dict[getattr(leaf2, self.attr)]
                if leaf1_idx == leaf2_idx:
                    # Instead of distance to itself, set this to be pendant edge length
                    mrca_matrix[leaf1_idx, leaf2_idx] = leaf1.dist + perturb
                    continue
                elif (mrca_matrix[leaf1_idx, leaf2_idx] + mrca_matrix[leaf2_idx, leaf1_idx]) > 0:
                    # We already filled this distance out
                    continue
                mrca = leaf1.get_common_ancestor(leaf2)
                mrca_dist = leaf1.get_distance(mrca)
                mrca_matrix[min(leaf1_idx, leaf2_idx), max(leaf1_idx, leaf2_idx)] = mrca_dist + perturb
                mrca_matrix[max(leaf1_idx, leaf2_idx), min(leaf1_idx, leaf2_idx)] = 0
        return mrca_matrix

    def get_dist(self, tree, C=0.1):
        tree_mrca_matrix = self._get_mrca_matrix(tree)
        difference_matrix = np.abs((self.ref_tree_mrca_matrix) - (tree_mrca_matrix))
        norm_diff = np.sum(difference_matrix)
        num_entries = (self.num_leaves - 1) * self.num_leaves / 2 + self.num_leaves
        return norm_diff/(num_entries)

class MRCASpearmanMeasurer(MRCADistanceMeasurer):
    """
    Using rank correlation of the MRCA distance matrix
    This is an adhoc measure, but maybe provides a good idea
    """
    name = "mrca_spearman"
    @staticmethod
    def get_upper_half(matrix):
        m_shape = matrix.shape[0]
        return np.array([matrix[i,j] for i in range(m_shape) for j in range(i, m_shape)])

    def get_dist(self, tree, collapse_thres: float = 0):
        raw_tree = tree.copy()
        # Collapse distances is requested
        for node in raw_tree.traverse("preorder"):
            if node.is_root():
                continue
            if node.dist < collapse_thres:
                old_dist = node.dist
                node.dist = 0
                for child in node.children:
                    child.dist += old_dist
        col_tree = collapsed_tree.collapse_zero_lens(raw_tree)

        tree_mrca_matrix = self._get_mrca_matrix(col_tree)

        tree_mrca_matrix_half = MRCASpearmanMeasurer.get_upper_half(tree_mrca_matrix)
        ref_mrca_matrix_half = MRCASpearmanMeasurer.get_upper_half(self.ref_tree_mrca_matrix)
        rank_corr, _ = kendalltau(tree_mrca_matrix_half, ref_mrca_matrix_half)
        #rank_corr, _ = spearmanr(tree_mrca_matrix_half, ref_mrca_matrix_half)
        return rank_corr


class InternalCorrMeasurer(MRCADistanceMeasurer):
    name = "internal_pearson"

    def __init__(
            self,
            ref_tree: CellLineageTree,
            scratch_dir: str = None,
            attr="allele_events_list_str",
            corr_func=pearsonr):
        self.ref_tree = ref_tree
        self.attr = attr
        self.corr_func = pearsonr

        # Number the leaves because we need to represent each tree by its pairwise
        # MRCA distance matrix
        leaf_str_sorted = [getattr(leaf, attr) for leaf in ref_tree]
        self.leaf_dict = {}
        for idx, leaf_str in enumerate(leaf_str_sorted):
            self.leaf_dict[leaf_str] = idx
        self.num_leaves = len(ref_tree)

        self.ref_tree_mrca_matrix = self._get_mrca_matrix(ref_tree)

        self.ref_internal_nodes, self.ref_node_val = self.get_ref_node_distances(ref_tree, self.ref_tree_mrca_matrix)

    def get_ref_node_distances(self, tree, tree_mrca_matrix):
        internal_nodes = [
                node for node in tree.traverse("preorder") if not node.is_root() and not node.is_leaf()]
        node_val = self._get_internal_node_dists(internal_nodes, tree_mrca_matrix)
        return internal_nodes, node_val

    def _get_internal_node_dists(self, internal_nodes, tree_mrca_matrix):
        node_val = []
        for node in internal_nodes:
            leaf_idxs = np.array([self.leaf_dict[getattr(leaf, self.attr)] for leaf in node])
            node_dist = np.max(tree_mrca_matrix[leaf_idxs, leaf_idxs])
            node_val.append(node_dist)
        return np.array(node_val)

    def _get_node_val(self, tree):
        tree_mrca_matrix = self._get_mrca_matrix(tree)
        return self._get_internal_node_dists(self.ref_internal_nodes, tree_mrca_matrix)

    def get_dist(self, tree):
        tree_mrca_matrix = self._get_mrca_matrix(tree)
        tree_node_val1 = self._get_internal_node_dists(self.ref_internal_nodes, tree_mrca_matrix)
        corr1, _ = self.corr_func(self.ref_node_val, tree_node_val1)

        tree_internal_nodes2, tree_node_val2 = self.get_ref_node_distances(tree, tree_mrca_matrix)
        ref_node_val2 = self._get_internal_node_dists(
                tree_internal_nodes2,
                self.ref_tree_mrca_matrix)
        corr2, _ = self.corr_func(ref_node_val2, tree_node_val2)

        logging.info("%f, %f", corr1, corr2)
        return 1 - (corr1 + corr2)/2


class InternalHeightsMeasurer(InternalCorrMeasurer):
    name = "internal_heights"

    def get_dist(self, tree):
        # TODO: SUPER JENK CODE
        tree_mrca_matrix = self._get_mrca_matrix(tree)
        tree_node_val1 = self.get_compare_node_distances(
                self.ref_internal_nodes,
                tree_mrca_matrix)
        height_dist1 = np.linalg.norm(self.ref_node_val - tree_node_val1)

        tree_internal_nodes2, tree_node_val2 = self.get_ref_node_distances(
                tree,
                tree_mrca_matrix)
        ref_node_val2 = self.get_compare_node_distances(
                tree_internal_nodes2,
                self.ref_tree_mrca_matrix)
        height_dist2 = np.linalg.norm(ref_node_val2 - tree_node_val2)

        logging.info("hiehg dist %s, %s", height_dist1, height_dist2)
        return (height_dist1 + height_dist2)/2

    def get_dist_multifurc_tree(self, tree, multifurc_tree):
        multifurc_leaf_sets = set()
        multifurc_tree_nodes = []
        for node in multifurc_tree.traverse("preorder"):
            leaf_tuple = tuple(l.allele_events_list_str for l in node)
            if leaf_tuple in multifurc_leaf_sets or len(leaf_tuple) <= 1:
                continue
            else:
                multifurc_leaf_sets.add(leaf_tuple)
                print(leaf_tuple)
                multifurc_tree_nodes.append(node)

        ref_node_dists = self._get_internal_node_dists(multifurc_tree_nodes, self.ref_tree_mrca_matrix)

        tree_mrca_matrix = self._get_mrca_matrix(tree)
        tree_node_dists = self._get_internal_node_dists(multifurc_tree_nodes, tree_mrca_matrix)

        print(ref_node_dists.size, tree_node_dists.size)

        return np.linalg.norm(ref_node_dists - tree_node_dists)
