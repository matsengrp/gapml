from numpy import ndarray
import numpy as np
from scipy.stats import expon
from numpy.random import choice, random
from scipy.stats import nbinom

from allele import Allele
from allele_simulator import AlleleSimulator
from barcode_metadata import BarcodeMetadata

class AlleleSimulatorSimultaneous(AlleleSimulator):
    def __init__(self,
        bcode_meta: BarcodeMetadata,
        target_lambdas: ndarray,
        left_long_prob: float,
        right_long_prob: float,
        del_probability: float,
        insert_probability: float,
        left_del_lambda: float,
        right_del_lambda: float,
        insertion_lambda: float):
        """
        @param bcode_meta: metadata about the barcode
        @param target_lambdas: rate parameter of each target in the allele
        @param del_probability: the probability of making an deletion
        @param insert_probability: the probability of making an insertion
        @param left_del_lambda
        @param right_del_lambda
        @param insertion_lambda
        """
        self.target_lambdas = target_lambdas
        self.left_long_prob = left_long_prob
        self.right_long_prob = right_long_prob
        self.del_probability = del_probability
        self.insert_probability = insert_probability

        self.left_del_distribution  = self.my_nbinom(left_del_mu,  left_del_alpha)
        self.right_del_distribution = self.my_nbinom(right_del_mu, right_del_alpha)
        self.insertion_distribution = self.my_nbinom(insertion_mu, insertion_alpha)

    def _race_target_cutting(self, allele: Allele):
        """
        Race target cutting (with no regard to time limits)
        @return race_winner:if a target cut won, then returns index of target
                            if no event happens (can't cut), then returns None
                event_time: the time of the event that won
                            if no event happens, then returns None
        """
        active_targets = allele.get_active_targets()
        num_active_targets = len(active_targets)
        if num_active_targets:
            target_cut_times = [
                expon.rvs(scale=1.0 / self.target_lambdas[i])
                for i in active_targets
            ]
            race_winner = active_targets[np.argmin(target_cut_times)]
            return race_winner, target_min_time
        else:
            # Nothing to cut and nothing to repair
            return None, None


    def simulate(self, init_allele: Allele, time: float):
        """
        @param init_allele: the initial state of the allele
        @param time: the amount of time to simulate the allele modification process

        @return allele after the simulation procedure
                    does not modify the allele that got passed in :)
        """
        assert(init_allele.n_targets == self.target_lambdas.size)

        allele = Allele(
            init_allele.allele,
            init_allele.unedited_allele,
            init_allele.cut_sites)

        time_remain = time
        while time_remain > 0:
            race_winner, event_time = self._race_target_cutting(allele)
            if race_winner is None:
                # Nothing to cut
                # no point in simulating the allele process then
                return allele
            time_remain = max(time_remain - event_time, 0)

            if time_remain > 0:
                # One of the targets got cut
                allele.cut(race_winner)
                self._do_repair(allele)
        return allele

    def _do_repair(self, allele: Allele):
        """
        Repairs allele.
        Does focal deletion if only one cut.
        Does inter-target deletion if 2 cuts.
        """
        if len(allele.needs_repair) not in (1, 2):
            raise ValueError('allele contains {} cuts, cannot repair'.format(len(allele.needs_repair)))
        target1 = min(allele.needs_repair)
        target2 = max(allele.needs_repair)

        # Serves for a zero-inflated negative binomial for deletion/insertion process
        # Draw a separate RVs for each deletion/insertion process
        do_insertion = random() < self.indel_probability
        do_deletion  = random() < self.indel_probability

        insertion_length = self.insertion_distribution.rvs() if do_insertion else 0
        left_del_len     = self.left_del_distribution.rvs() if do_deletion else 0
        right_del_len    = self.right_del_distribution.rvs() if do_deletion else 0

        # TODO: make this more realistic. right now just random DNA inserted
        insertion = ''.join(choice(list('acgt'), insertion_length))
        allele.indel(
            target1,
            target2,
            left_del_len,
            right_del_len,
            insertion)
