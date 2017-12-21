from typing import List, Dict, Tuple
import numpy as np

from barcode import Barcode
from cell_state import CellState
from cell_lineage_tree import CellLineageTree
from alignment import Aligner

class ObservedAlignedSeq:
    def __init__(self,
            barcode: Barcode,
            events: List[Tuple[int, int, str]],
            cell_state: CellState,
            abundance: float):
        """
        Stores barcodes that are observed.
        Since these are stored using the barcode class,
        it implicitly gives us the alignment
        """
        self.barcode = barcode
        self.events = events
        self.cell_state = cell_state
        self.abundance = abundance


class CLTObserver:
    def __init__(self, sampling_rate: float,
                 error_rate: float = 0,
                 aligner: Aligner = None):
        """
        @param sampling_rate: the rate at which barcodes from the alive leaf cells are observed
        @param error_rate: sequencing error, introduce alternative bases uniformly at this rate
        """
        assert (0 < sampling_rate <= 1)
        assert (0 <= error_rate <= 1)
        self.sampling_rate = sampling_rate
        self.error_rate = error_rate
        self.aligner = aligner

    def observe_leaves(self,
                       cell_lineage_tree: CellLineageTree,
                       give_pruned_clt: bool = True):
        """
        Samples leaves from the cell lineage tree, of those that are not dead

        TODO: this probably won't work very well for very large trees.

        @param cell_lineage_tree: tree to sample leaves from

        @return a list of the sampled observations (List[ObservedAlignedSeq])
                a cell lineage tree with pruned leaves
        """
        clt = cell_lineage_tree.copy()
        observations = {}
        observed_leaves = set()
        for leaf in clt:
            if not leaf.dead:
                is_sampled = np.random.binomial(1, self.sampling_rate)
                if is_sampled:
                    observed_leaves.add(leaf.name)
                    barcode_with_errors = leaf.barcode.observe_with_errors(self.error_rate)
                    barcode_with_errors_events = barcode_with_errors.get_events(aligner=self.aligner)
                    leaf.barcode = barcode_with_errors
                    cell_id = (str(barcode_with_errors_events), str(leaf.cell_state))
                    if cell_id in observations:
                        observations[cell_id].abundance += 1
                    else:
                        observations[cell_id] = ObservedAlignedSeq(
                            barcode=barcode_with_errors,
                            events=barcode_with_errors_events,
                            cell_state=leaf.cell_state,
                            abundance=1,
                        )

        if len(observations) == 0:
            raise RuntimeError('all lineages extinct, nothing to observe')

        if give_pruned_clt:
            for node in clt.iter_descendants():
                if sum((node2.name in observed_leaves) for node2 in node.traverse()) == 0:
                    node.detach()
            # remove remaining unifurcations
            for node in clt.iter_descendants():
                parent = node.up
                if len(node.children) == 1:
                    node.delete(prevent_nondicotomic=False, preserve_branch_length=True)
            assert sum(leaf.abundance for leaf in observations.values()) == len(clt)
            return list(observations.values()), clt
        else:
            return list(observations.values())
