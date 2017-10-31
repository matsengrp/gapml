from numpy import ndarray
import numpy as np
from scipy.stats import expon, poisson, binom
from numpy.random import choice

from barcode import Barcode


class BarcodeSimulator:
    """
    Simulate how a barcode gets cut/repaired.
    Subclass this if you want to experiment with other models of the barcode cut/repair process.

    This simulator assumes each barcode can have at most two cuts.
    """

    def __init__(self,
        target_lambdas: ndarray,
        repair_rates: ndarray,
        indel_probability: float,
        left_del_lambda: float,
        right_del_lambda: float,
        insertion_lambda: float):
        """
        @param target_lambdas: rate parameter of each target in the barcode
        @param repair_rates: rate parameter of repair for N cuts in the barcode (repair_rates[N - 1] = rate with N cut in the barcode)
        @param indel_probability: the probability of making an insertion/deletion (currently this zero-inflation parameter is shared)
        @param left_del_lambda: poisson parameter for number of nucleotides deleted to the left of the DSB
        @param right_del_lambda: poisson parameter for number of nucleotides deleted to the right of the DSB
        @param insertion_lambda: poisson parameter for number of nucleotides insertd after DSB

        # TODO: inflation probability parameter for indels is shared. maybe not realistic
        """
        self.target_lambdas = target_lambdas
        self.repair_rates = repair_rates
        self.indel_probability = indel_probability
        self.left_del_lambda = left_del_lambda
        self.right_del_lambda = right_del_lambda
        self.insertion_lambda = insertion_lambda

    def _race_repair_target_cutting(self, barcode: Barcode):
        """
        Race repair process and target cutting (with no regard to time limits)
        @return race_winner: if a target cut won, then returns index of target
                            if a repair won, then returns -1
                            if no event happens (can't repair and can't cut), then returns None
                event_time: the time of the event that won
                            if no event happens, then returns None
        """
        num_needs_repair = len(barcode.needs_repair)
        repair_scale = 1.0 / self.repair_rates[num_needs_repair - 1]
        repair_time = expon.rvs(scale=repair_scale) if num_needs_repair else np.inf
        active_targets = barcode.get_active_targets()
        num_active_targets = len(active_targets)
        if num_needs_repair >= 2 or (num_needs_repair >= 1 and num_active_targets == 0):
            # Assumes no more than two cuts in barcode
            # The other case is that there is nothing to cut
            return -1, repair_time
        elif num_active_targets > 0:
            # If barcode has <=1 cuts, then barcode may get cut
            target_cut_times = [
                expon.rvs(scale=1.0 / self.target_lambdas[i])
                for i in active_targets
            ]
            target_min_time = np.min(target_cut_times)
            if repair_time < target_min_time:
                # Repair process won
                return -1, repair_time
            else:
                # Target cutting won
                race_winner = active_targets[np.argmin(target_cut_times)]
                return race_winner, target_min_time
        else:
            # Nothing to cut and nothing to repair
            return None, None


    def simulate(self, init_barcode: Barcode, time: float):
        """
        @param init_barcode: the initial state of the barcode
        @param time: the amount of time to simulate the barcode modification process

        @return barcode after the simulation procedure
                    does not modify the barcode that got passed in :)
        """
        assert(init_barcode.n_targets == self.target_lambdas.size)

        barcode = Barcode(
            init_barcode.barcode,
            init_barcode.unedited_barcode,
            init_barcode.cut_sites)

        time_remain = time
        while time_remain > 0:
            race_winner, event_time = self._race_repair_target_cutting(barcode)
            if race_winner is None:
                # Nothing to repair and nothing to cut
                # no point in simulating the barcode process then
                return barcode
            time_remain = max(time_remain - event_time, 0)

            if time_remain > 0 and race_winner >= 0:
                # One of the targets got cut
                barcode.cut(race_winner)
                continue

            # Do repair if one of the two bool flags is true:
            ## (1) If barcode is still broken but we ran out of time, make sure we fix the barcode.
            ## (2) the repair process won the race so we just repair the barcode.
            self._do_repair(barcode)
        return barcode

    def _do_repair(self, barcode: Barcode):
        """
        Repairs barcode.
        Does focal deletion if only one cut.
        Does inter-target deletion if 2 cuts.
        """
        target1 = min(barcode.needs_repair)
        target2 = max(barcode.needs_repair)

        # Serves for a zero-inflated poisson for deletion/insertion process
        # Draw a separate RVs for each deletion/insertion process
        indel_action = binom.rvs(n=1, p=self.indel_probability, size=3)

        left_del_len = poisson.rvs(
            self.left_del_lambda) if indel_action[0] else 0
        right_del_len = poisson.rvs(
            self.right_del_lambda) if indel_action[1] else 0
        insertion_length = poisson.rvs(
            self.insertion_lambda) if indel_action[2] else 0

        # TODO: make this more realistic. right now just random DNA inserted
        insertion = ''.join(choice(list('acgt'), insertion_length))
        barcode.indel(
            target1,
            target2,
            left_del_len,
            right_del_len,
            insertion)
