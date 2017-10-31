import numpy as np

from barcode import Barcode
from cell_state import CellState
from cell_lineage_tree import CellLineageTree


class ObservedAlignedSeq:
    def __init__(self,
            barcode: Barcode,
            cell_state: CellState,
            abundance: float):
        """
        Stores barcodes that are observed.
        Since these are stored using the barcode class,
        it implicitly gives us the alignment
        """
        self.barcode = barcode
        self.cell_state = cell_state
        self.abundance = abundance


class CLTObserver:
    def __init__(self, sampling_rate: float):
        """
        @param sampling_rate: the rate at which barcodes from the alive leaf cells are observed
        """
        assert (sampling_rate <= 1 and sampling_rate > 0)
        self.sampling_rate = sampling_rate

    def observe_leaves(self, cell_lineage_tree: CellLineageTree):
        """
        Samples leaves from the cell lineage tree, of those that are not dead

        TODO: this probably won't work very well for very large trees.

        @param cell_lineage_tree: tree to sample leaves from

        @return a list of the sampled observations (List[ObservedAlignedSeq])
                a cell lineage tree with pruned leaves
        """
        clt = cell_lineage_tree.copy()
        observations = {}
        observed_leaves = []
        for leaf in cell_lineage_tree:
            if not leaf.dead:
                is_sampled = np.random.binomial(1, self.sampling_rate)
                if is_sampled:
                    observed_leaves.append(leaf.name)
                    cell_id = (str(leaf.barcode), str(leaf.cell_state))
                    if cell_id in observations:
                        observations[cell_id].abundance += 1
                    else:
                        observations[cell_id] = ObservedAlignedSeq(
                            barcode=leaf.barcode,
                            cell_state=leaf.cell_state,
                            abundance=1,
                        )

        clt.prune(observed_leaves)
        return list(observations.values()), clt
