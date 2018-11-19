from numpy import ndarray
from typing import List
import numpy as np
from scipy.stats import expon
from numpy.random import choice, random
from scipy.stats import nbinom

from allele import Allele, AlleleList
from indel_sets import TargetTract
from cell_lineage_tree import CellLineageTree
from allele_simulator import AlleleSimulator
from bounded_distributions import ZeroInflatedBoundedNegativeBinomial, PaddedBoundedNegativeBinomial

from clt_likelihood_model import CLTLikelihoodModel
from common import sigmoid

class AlleleSimulatorSimultaneous(AlleleSimulator):
    """
    Allele cut/repair simulator where the cut/repair are simultaneous
    """
    def __init__(self,
        model: CLTLikelihoodModel,
        boost_len: int = 1):
        """
        @param model
        @param boost_probs: a 3-dim array that indicates the probability of a boost in length
                        being applied to the insertion, left del, right del distributions.
                        The boost will shift the insertion dist by `boost_len`.
                        The boost will change the minimum left del len to `boost_len` (but doesn't
                        shift the max length -- nbinom distributions are properly normalized).
                        Likewise for the right del dist.
                        Note that boosts are not applied if the left or right deletions are long.
        @param boost_len: the amount to boost up the length of the indel
        """
        self.bcode_meta = model.bcode_meta
        self.model = model
        self.all_target_tract_hazards = model.get_all_target_tract_hazards()
        self.insert_zero_prob = self.model.insert_zero_prob.eval()
        self.trim_zero_probs = self.model.trim_zero_probs.eval()

        assert boost_len > 0
        self.boost_probs = self.model.boost_probs.eval()
        self.boost_len = boost_len

        self.left_del_distributions = self._create_bounded_nbinoms(
            min_vals = self.bcode_meta.left_long_trim_min,
            max_vals = self.bcode_meta.left_max_trim,
            short_nbinom_m = self.model.trim_short_nbinom_m[0].eval(),
            short_nbinom_logit = self.model.trim_short_nbinom_logits[0].eval(),
            long_nbinom_m = self.model.trim_long_nbinom_m[0].eval(),
            long_nbinom_logit = self.model.trim_long_nbinom_logits[0].eval())
        self.right_del_distributions = self._create_bounded_nbinoms(
            min_vals = self.bcode_meta.right_long_trim_min,
            max_vals = self.bcode_meta.right_max_trim,
            short_nbinom_m = self.model.trim_short_nbinom_m[1].eval(),
            short_nbinom_logit = self.model.trim_short_nbinom_logits[1].eval(),
            long_nbinom_m = self.model.trim_long_nbinom_m[1].eval(),
            long_nbinom_logit = self.model.trim_long_nbinom_logits[1].eval())
        self.insertion_distribution = nbinom(
                self.model.insert_nbinom_m.eval(),
                1 - sigmoid(self.model.insert_nbinom_logit.eval()))

    def get_root(self):
        return AlleleList(
                [self.bcode_meta.unedited_barcode] * self.bcode_meta.num_barcodes,
                self.bcode_meta)

    def _create_bounded_nbinoms(self,
            min_vals: List[float],
            max_vals: List[float],
            short_nbinom_m: float,
            short_nbinom_logit: float,
            long_nbinom_m: float,
            long_nbinom_logit: float):
        """
        @param min_vals: the min long trim length for this target (left or right)
        @param max_vals: the max trim length for this target (left or right)
        @param short_binom_m: neg binom parameter for short trims, number of failures
        @param short_binom_logit: neg binom parameter for short trims, logit for prob of success
        @param long_binom_m: neg binom parameter for long trims, number of failures
        @param long_binom_logit: neg binom parameter for long trims, logit for prob of success

        @return bounded poisson distributions for each target, for long, short , boosted-short trims
                List[Dict[bool, BoundedPoisson]]
        """
        dstns = []
        for i in range(self.bcode_meta.n_targets):
            long_short_dstns = {
                    # Long trim
                    "long": PaddedBoundedNegativeBinomial(min_vals[i], max_vals[i], long_nbinom_m, long_nbinom_logit),
                    # Short trim
                    "short": ZeroInflatedBoundedNegativeBinomial(0, min_vals[i] - 1, short_nbinom_m, short_nbinom_logit),
                    # Boosted short trim
                    "boost_short": ZeroInflatedBoundedNegativeBinomial(self.boost_len, min_vals[i] - 1, short_nbinom_m, short_nbinom_logit)}
            dstns.append(long_short_dstns)
        return dstns

    def _race_target_tracts(self, allele: Allele, scale_hazard: float):
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
                self.all_target_tract_hazards[self.model.target_tract_dict[tt]] * scale_hazard
                for tt in target_tracts]
            all_haz_sum = np.sum(all_hazards)
            min_time = expon.rvs(scale=1.0/all_haz_sum)
            race_winner = target_tracts[
                np.random.choice(len(target_tracts), p=np.array(all_hazards)/all_haz_sum)]
            return race_winner, min_time
        else:
            return None, None

    def simulate(self,
            init_allele: Allele,
            node: CellLineageTree,
            scale_hazard_func = lambda x: 1,
            time_incr: float = 0.025):
        """
        @param init_allele: the initial state of the allele
        @param scale_hazard_func: a function that takes in the current time in the tree
                        and returns how much to scale the cut rates

        @return allele after the simulation procedure
        """
        allele = Allele(init_allele.allele, init_allele.bcode_meta)

        # TODO: this is currently the stupidest and slowest implementation of changing target cut rates
        # try to make this faster please.
        time_remain = node.dist
        while time_remain > 0:
            scale_hazard = scale_hazard_func(node.up.dist_to_root + node.dist - time_remain)
            target_tract, event_time = self._race_target_tracts(allele, scale_hazard)
            if event_time is None:
                break

            if event_time < min(time_remain, time_incr):
                # Target(s) got cut
                self._do_repair(allele, target_tract)
                time_remain -= event_time
            else:
                time_remain -= time_incr

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

        left_short_boost = 0
        right_short_boost = 0
        insert_boost = 0
        left_distr_key = "long" if left_long else "short"
        right_distr_key = "long" if right_long else "short"
        do_insertion = random() > self.insert_zero_prob
        if left_long or right_long:
            # No zero inflation if we decided to do a long left or right trim
            # No boosts if long left or right del
            do_deletion = [left_long, right_long]
        else:
            # Determine whether to boost insert vs left del vs right del
            len_incr_rv = np.random.multinomial(n=1, pvals=self.boost_probs)

            if len_incr_rv[0] == 1:
                insert_boost = self.boost_len
            elif len_incr_rv[1] == 1:
                left_short_boost = self.boost_len
            elif len_incr_rv[2] == 1:
                right_short_boost = self.boost_len

            # Serves as zero-inflation for deletion/insertion process
            # Draw a separate RVs for each deletion/insertion process
            do_deletion = random(2) > self.trim_zero_probs

        if insert_boost:
            insertion_length = insert_boost + self.insertion_distribution.rvs()
        else:
            insertion_length = self.insertion_distribution.rvs() if do_insertion else 0
        if left_short_boost:
            left_del_len = self.left_del_distributions[target1]["boost_short"].rvs()
        else:
            left_del_len = self.left_del_distributions[target1][left_distr_key].rvs() if do_deletion[0] else 0
        if right_short_boost:
            right_del_len = self.right_del_distributions[target2]["boost_short"].rvs()
        else:
            right_del_len = self.right_del_distributions[target2][right_distr_key].rvs() if do_deletion[1] else 0

        # TODO: make this more realistic. right now just random DNA inserted
        insertion = ''.join(choice(list('acgt'), insertion_length))
        allele.indel(
            target1,
            target2,
            left_del_len,
            right_del_len,
            insertion)
