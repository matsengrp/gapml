from numpy import ndarray
import numpy as np
from scipy.stats import expon
from numpy.random import choice, random
from scipy.stats import nbinom

from allele import Allele
from allele_simulator import AlleleSimulator


class AlleleSimulatorCutRepair(AlleleSimulator):
    """
    Simulate how a allele gets cut/repaired.
    Cut and repair is not simultaneous.
    Subclass this if you want to experiment with other models of the allele cut/repair process.

    This simulator assumes each allele can have at most two cuts.
    """

    @staticmethod
    def my_nbinom(mu: float, alpha: float):
        """
        reparameterize negative binomial in terms of mean and dispersion
        @param mu: mean of negative binomial
        @param alpha: dispersion of negative binomial
        @return corresponding scipy.stats.nbinom
        """
        p = 1 - mu / (mu + 1 / alpha)
        n = 1 / alpha
        return nbinom(n, p)

    def __init__(self,
        target_lambdas: ndarray,
        repair_rates: ndarray,
        indel_probability: float,
        left_del_mu: float,
        right_del_mu: float,
        insertion_mu: float,
        left_del_alpha: float = 1.,
        right_del_alpha: float = 1.,
        insertion_alpha: float = 1.):
        """
        @param target_lambdas: rate parameter of each target in the allele
        @param repair_rates: rate parameter of repair for N cuts in the allele (repair_rates[N - 1] = rate with N cut in the allele)
        @param indel_probability: the probability of making an insertion/deletion (currently this zero-inflation parameter is shared)
        @param left_del_mu: NB mean for number of nucleotides deleted to the left of the DSB
        @param right_del_mu: NB mean for number of nucleotides deleted to the right of the DSB
        @param insertion_mu: NB mean for number of nucleotides insertd after DSB
        @param left_del_alpha: NB dispersion for number of nucleotides deleted to the left of the DSB
        @param right_del_alpha: NB dispersion for number of nucleotides deleted to the right of the DSB
        @param insertion_alpha: NB dispersion for number of nucleotides insertd after DSB

        # TODO: inflation probability parameter for indels is shared. maybe not realistic
        """
        self.target_lambdas = target_lambdas
        self.repair_rates = repair_rates
        self.indel_probability = indel_probability

        self.left_del_distribution  = self.my_nbinom(left_del_mu,  left_del_alpha)
        self.right_del_distribution = self.my_nbinom(right_del_mu, right_del_alpha)
        self.insertion_distribution = self.my_nbinom(insertion_mu, insertion_alpha)

    def _race_repair_target_cutting(self, allele: Allele):
        """
        Race repair process and target cutting (with no regard to time limits)
        @return race_winner: if a target cut won, then returns index of target
                            if a repair won, then returns -1
                            if no event happens (can't repair and can't cut), then returns None
                event_time: the time of the event that won
                            if no event happens, then returns None
        """
        num_needs_repair = len(allele.needs_repair)
        repair_scale = 1.0 / self.repair_rates[num_needs_repair - 1]
        repair_time = expon.rvs(scale=repair_scale) if num_needs_repair else np.inf
        active_targets = allele.get_active_targets()
        num_active_targets = len(active_targets)
        if num_needs_repair >= 2 or (num_needs_repair >= 1 and num_active_targets == 0):
            # Assumes no more than two cuts in allele
            # The other case is that there is nothing to cut
            return -1, repair_time
        elif num_active_targets > 0:
            # If allele has <=1 cuts, then allele may get cut
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


    def simulate(self, init_allele: Allele, time: float):
        """
        @param init_allele: the initial state of the allele
        @param time: the amount of time to simulate the allele modification process

        @return allele after the simulation procedure
                    does not modify the allele that got passed in :)
        """
        allele = Allele(
            init_allele.allele,
            init_allele.bcode_meta)

        time_remain = time
        while time_remain > 0:
            race_winner, event_time = self._race_repair_target_cutting(allele)
            if race_winner is None:
                # Nothing to repair and nothing to cut
                # no point in simulating the allele process then
                return allele
            time_remain = max(time_remain - event_time, 0)

            if time_remain > 0 and race_winner >= 0:
                # One of the targets got cut
                allele.cut(race_winner)
                continue

            # Do repair if one of the two bool flags is true:
            ## (1) If allele is still broken but we ran out of time, make sure we fix the allele.
            ## (2) the repair process won the race so we just repair the allele.
            if (time_remain == 0 and len(allele.needs_repair) > 0) or race_winner == -1:
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
