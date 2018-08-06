from numpy import ndarray
from typing import List
import numpy as np
from scipy.stats import expon
from numpy.random import choice, random
from scipy.stats import poisson

from allele import Allele, AlleleList
from indel_sets import TargetTract
from target_status import TargetStatus
from allele_simulator import AlleleSimulator
from barcode_metadata import BarcodeMetadata
from bounded_poisson import ZeroInflatedBoundedPoisson, PaddedBoundedPoisson

from clt_likelihood_model import CLTLikelihoodModel

class AlleleSimulatorSimultaneous(AlleleSimulator):
    """
    Allele cut/repair simulator where the cut/repair are simultaneous
    """
    def __init__(self,
        model: CLTLikelihoodModel,
        boost_len: int = 1):
        """
        @param model
        """
        self.bcode_meta = model.bcode_meta
        self.model = model
        self.all_target_tract_hazards = model.get_all_target_tract_hazards()
        self.insert_zero_prob = self.model.insert_zero_prob.eval()
        self.trim_zero_probs = self.model.trim_zero_probs.eval()
        self.boost_len = boost_len

        self.left_del_distributions = self._create_bounded_poissons(
            min_vals = self.bcode_meta.left_long_trim_min,
            max_vals = self.bcode_meta.left_max_trim,
            poiss_short = self.model.trim_short_poissons[0].eval(),
            poiss_long = self.model.trim_long_poissons[0].eval(),
            boost_len = boost_len)
        self.right_del_distributions = self._create_bounded_poissons(
            min_vals = self.bcode_meta.right_long_trim_min,
            max_vals = self.bcode_meta.right_max_trim,
            poiss_short = self.model.trim_short_poissons[1].eval(),
            poiss_long = self.model.trim_long_poissons[1].eval(),
            boost_len = boost_len)
        self.insertion_distribution = poisson(mu=self.model.insert_poisson.eval())

    def get_root(self):
        return AlleleList(
                [self.bcode_meta.unedited_barcode] * self.bcode_meta.num_barcodes,
                self.bcode_meta)

    def _create_bounded_poissons(self,
            min_vals: List[float],
            max_vals: List[float],
            poiss_short: float,
            poiss_long: float,
            boost_len: int = 1):
        """
        @param min_vals: the min long trim length for this target (left or right)
        @param max_vals: the max trim length for this target (left or right)
        @param poisson_short: poisson parameter for short trims
        @param poisson_long: poisson parameter for long trims

        @return bounded poisson distributions for each target, for long and short trims
                List[Dict[bool, BoundedPoisson]]
        """
        dstns = []
        for i in range(self.bcode_meta.n_targets):
            long_short_dstns = {
                    # Long trim
                    "long": PaddedBoundedPoisson(min_vals[i], max_vals[i], poiss_long),
                    # Short trim
                    "short": ZeroInflatedBoundedPoisson(0, min_vals[i] - 1, poiss_short),
                    # Boosted short trim
                    "boost_short": ZeroInflatedBoundedPoisson(boost_len, min_vals[i] - 1, poiss_short)}
            dstns.append(long_short_dstns)
        return dstns

    def _race_target_tracts(self, allele: Allele):
        """
        Race cutting (with no regard to time limits)
        @return race_winner: target tract if event occurs
                            if no event happens (can't cut), then returns None
                event_time: the time of the event that won
                            if no event happens, then returns None
        """
        targ_stat = allele.get_target_status()
        target_tracts = targ_stat.get_possible_target_tracts(self.bcode_meta)
        if len(target_tracts):
            all_hazards = [
                self.all_target_tract_hazards[self.model.target_tract_dict[tt]]
                for tt in target_tracts]
            all_haz_sum = np.sum(all_hazards)
            min_time = expon.rvs(scale=1.0/all_haz_sum)
            race_winner = target_tracts[
                np.random.choice(len(target_tracts), p=np.array(all_hazards)/all_haz_sum)]
            return race_winner, min_time
        else:
            return None, None

    def simulate(self, init_allele: Allele, time: float):
        """
        @param init_allele: the initial state of the allele
        @param time: the amount of time to simulate the allele modification process

        @return allele after the simulation procedure
        """
        allele = Allele(init_allele.allele, init_allele.bcode_meta)

        time_remain = time
        while time_remain > 0:
            target_tract, event_time = self._race_target_tracts(allele)
            if event_time is None:
                break
            time_remain = max(time_remain - event_time, 0)

            if time_remain > 0:
                # Target(s) got cut
                self._do_repair(allele, target_tract)
        return allele

    def _do_repair(self, allele: Allele, target_tract: TargetTract):
        """
        Repairs allele per the target_tract
        NOTE: if this tries to delete an already-deleted position,
              this simulation will keep the next non-deleted position
        """
        target1 = target_tract.min_target
        target2 = target_tract.max_target

        left_long = target_tract.is_left_long
        right_long = target_tract.is_right_long

        insert_boost = 0
        left_short_boost = 0
        right_short_boost = 0
        left_distr_key = "long" if left_long else "short"
        right_distr_key = "long" if right_long else "short"

        do_insertion = random() > self.insert_zero_prob
        if left_long or right_long:
            # No zero inflation if we decided to do a long left or right trim
            do_deletion = [True, True]
        else:
            len_incr_rv = np.random.multinomial(1, [1/3.] * 3)
            if len_incr_rv[0] == 1:
                insert_boost = self.boost_len
            elif len_incr_rv[1] == 1:
                left_distr_key = "boost_short"
                left_short_boost = self.boost_len
            else:
                right_distr_key = "boost_short"
                right_short_boost = self.boost_len

            # Serves as zero-inflation for deletion/insertion process
            # Draw a separate RVs for each deletion/insertion process
            do_deletion  = random(2) > self.trim_zero_probs

        insertion_length = insert_boost + (self.insertion_distribution.rvs() if do_insertion else 0)
        left_del_len = self.left_del_distributions[target1][left_distr_key].rvs() if do_deletion[0] else left_short_boost
        right_del_len = self.right_del_distributions[target2][right_distr_key].rvs() if do_deletion[1] else right_short_boost

        # TODO: make this more realistic. right now just random DNA inserted
        insertion = ''.join(choice(list('acgt'), insertion_length))
        allele.indel(
            target1,
            target2,
            left_del_len,
            right_del_len,
            insertion)
