from numpy import ndarray
from typing import List
import numpy as np
from scipy.stats import expon
from numpy.random import choice, random
from scipy.stats import poisson

from allele import Allele, AlleleList
from indel_sets import TargetTract
from allele_simulator import AlleleSimulator
from barcode_metadata import BarcodeMetadata
from bounded_poisson import BoundedPoisson

from clt_likelihood_model import CLTLikelihoodModel

class AlleleSimulatorSimultaneous(AlleleSimulator):
    """
    Allele cut/repair simulator where the cut/repair are simultaneous
    """
    def __init__(self,
        model: CLTLikelihoodModel):
        """
        @param model
        """
        self.bcode_meta = model.bcode_meta
        self.model = model

        self.left_del_distributions = self._create_bounded_poissons(
            min_vals = self.bcode_meta.left_long_trim_min,
            max_vals = self.bcode_meta.left_max_trim,
            poiss_lambda = self.model.trim_poissons[0].eval())
        self.right_del_distributions = self._create_bounded_poissons(
            min_vals = self.bcode_meta.right_long_trim_min,
            max_vals = self.bcode_meta.right_max_trim,
            poiss_lambda = self.model.trim_poissons[1].eval())
        self.insertion_distribution = poisson(mu=self.model.insert_poisson.eval())

    def get_root(self):
        return AlleleList(
                [self.bcode_meta.unedited_barcode] * self.bcode_meta.num_barcodes,
                self.bcode_meta)

    def _create_bounded_poissons(self, min_vals: List[float], max_vals: List[float], poiss_lambda: float):
        """
        @param min_vals: the min long trim length for this target (left or right)
        @param max_vals: the max trim length for this target (left or right)
        @param poisson_lambda: poisson parameter for long and short trims (right now they are the same)

        @return bounded poisson distributions for each target, for long and short trims
                List[Dict[bool, BoundedPoisson]]
        """
        dstns = []
        for i in range(self.bcode_meta.n_targets):
            long_min = min_vals[i]
            long_short_dstns = {}
            for is_long in [True, False]:
                min_trim = min_vals[i] if is_long else 0
                max_trim = max_vals[i] if is_long else min_vals[i] - 1

                dstn = BoundedPoisson(min_trim, max_trim, poiss_lambda)
                long_short_dstns[is_long] = dstn
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
        active_targets = allele.get_active_targets()
        num_active_targets = len(active_targets)
        if num_active_targets:
            target_tracts = list(CLTLikelihoodModel.get_possible_target_tracts(active_targets))
            all_hazards = self.model.get_hazards(target_tracts)
            tt_times = [expon.rvs(scale=1.0 / hz) for hz in all_hazards]
            race_winner = target_tracts[np.argmin(tt_times)]
            min_time = np.min(tt_times)
            return race_winner, min_time
        else:
            return None, None

    def simulate(self, init_allele: Allele, time: float):
        """
        @param init_allele: the initial state of the allele
        @param time: the amount of time to simulate the allele modification process

        @return allele list after the simulation procedure
                does not modify the allele that got passed in :)
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
                allele.cut(target_tract.min_target)
                if target_tract.min_target != target_tract.max_target:
                    allele.cut(target_tract.max_target)
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

        if left_long or right_long:
            # No zero inflation if we decided to do a long left or right trim
            do_insertion = True
            do_deletion = True
        else:
            # Serves as zero-inflation for deletion/insertion process
            # Draw a separate RVs for each deletion/insertion process
            do_insertion = random() > self.model.insert_zero_prob.eval()
            do_deletion  = random() > self.model.trim_zero_prob.eval()

        insertion_length = self.insertion_distribution.rvs() if do_insertion else 0
        left_del_len = self.left_del_distributions[target1][left_long].rvs() if do_deletion else 0
        right_del_len = self.right_del_distributions[target2][right_long].rvs() if do_deletion else 0
        if left_long:
            assert(left_del_len > 0)
        if right_long:
            assert(right_del_len > 0)

        # TODO: make this more realistic. right now just random DNA inserted
        insertion = ''.join(choice(list('acgt'), insertion_length))
        allele.indel(
            target1,
            target2,
            left_del_len,
            right_del_len,
            insertion)
