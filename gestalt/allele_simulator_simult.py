from typing import List
from numpy import ndarray
import numpy as np
from scipy.stats import expon
from numpy.random import choice, random
from scipy.stats import poisson, nbinom

from allele import Allele, AlleleList
from indel_sets import TargetTract
from cell_lineage_tree import CellLineageTree
from allele_simulator import AlleleSimulator
from bounded_distributions import PaddedBoundedPoisson
from bounded_distributions import ShiftedPoisson
from bounded_distributions import ConditionalBoundedNegativeBinomial
from bounded_distributions import ShiftedNegativeBinomial

from common import sigmoid
from clt_likelihood_model import CLTLikelihoodModel

class AlleleSimulatorSimultaneous(AlleleSimulator):
    """
    Allele cut/repair simulator where the cut/repair are simultaneous
    """
    def __init__(self, model: CLTLikelihoodModel):
        self.bcode_meta = model.bcode_meta
        self.model = model
        self.insert_zero_prob = self.model.insert_zero_prob.eval()
        self.trim_zero_prob_dict = self.model.trim_zero_prob_dict.eval()

        self.all_target_tract_hazards = model.get_all_target_tract_hazards()

        insert_params = self.model.insert_params.eval()
        if self.model.use_poisson:
            self.left_del_dist = self._create_bounded_poissons(
                min_vals = self.bcode_meta.left_long_trim_min,
                max_vals = self.bcode_meta.left_max_trim,
                short_poiss = self.model.trim_short_params[0].eval(),
                long_poiss = self.model.trim_long_params[0].eval())
            self.right_del_dist = self._create_bounded_poissons(
                min_vals = self.bcode_meta.right_long_trim_min,
                max_vals = self.bcode_meta.right_max_trim,
                short_poiss = self.model.trim_short_params[1].eval(),
                long_poiss = self.model.trim_long_params[1].eval())
            self.insertion_distribution = ShiftedPoisson(1, np.exp(insert_params[0]))
        else:
            self.left_del_dist = self._create_bounded_nbinoms(
                min_vals = self.bcode_meta.left_long_trim_min,
                max_vals = self.bcode_meta.left_max_trim,
                short_params = self.model.trim_short_params_reshaped[0,:].eval(),
                long_params = self.model.trim_long_params_reshaped[0,:].eval())
            self.right_del_dist = self._create_bounded_nbinoms(
                min_vals = self.bcode_meta.right_long_trim_min,
                max_vals = self.bcode_meta.right_max_trim,
                short_params = self.model.trim_short_params_reshaped[1,:].eval(),
                long_params = self.model.trim_long_params_reshaped[1,:].eval())
            self.insertion_distribution = ShiftedNegativeBinomial(1, np.exp(insert_params[0]), insert_params[1])

    def get_root(self):
        return AlleleList(
                [self.bcode_meta.unedited_barcode] * self.bcode_meta.num_barcodes,
                self.bcode_meta)

    def _create_bounded_nbinoms(self,
            min_vals: List[float],
            max_vals: List[float],
            short_params: ndarray,
            long_params: ndarray):
        """
        @param min_vals: the min long trim length for this target (left or right)
        @param max_vals: the max trim length for this target (left or right)
        @param short_binom_m: neg binom parameter for short trims, number of failures
        @param long_binom_m: neg binom parameter for long trims, number of failures

        @return bounded poisson distributions for each target, for long, short , boosted-short trims
                List[Dict[bool indicating long trim, BoundedNegativeBinomial]]
        """
        dstns = []
        for i in range(self.bcode_meta.n_targets):
            long_short_dstns = [
                    # Short trim
                    ConditionalBoundedNegativeBinomial(
                        1,
                        min_vals[i] - 1,
                        np.exp(short_params[0]),
                        short_params[1]),
                    # Long trim
                    ConditionalBoundedNegativeBinomial(
                        min_vals[i],
                        max_vals[i],
                        np.exp(long_params[0]),
                        long_params[1])]
            dstns.append(long_short_dstns)
        return dstns

    def _create_bounded_poissons(self,
            min_vals: List[float],
            max_vals: List[float],
            short_poiss: float,
            long_poiss: float):
        """
        @param min_vals: the min long trim length for this target (left or right)
        @param max_vals: the max trim length for this target (left or right)
        @param short_binom_m: neg binom parameter for short trims, number of failures
        @param long_binom_m: neg binom parameter for long trims, number of failures

        @return bounded poisson distributions for each target, for long, short , boosted-short trims
                List[Dict[bool indicating long trim, BoundedPoisson]]
        """
        dstns = []
        for i in range(self.bcode_meta.n_targets):
            long_short_dstns = [
                    # Short trim
                    PaddedBoundedPoisson(1, min_vals[i] - 1, np.exp(short_poiss)),
                    # Long trim
                    PaddedBoundedPoisson(min_vals[i], max_vals[i], np.exp(long_poiss))]
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
        active_targets = targ_stat.get_active_targets(self.bcode_meta)
        num_active = len(active_targets)
        if num_active == 0:
            race_winner = None
            min_time = None
        else:
            target_tracts = targ_stat.get_possible_target_tracts(self.bcode_meta)
            all_hazards = [
                self.all_target_tract_hazards[self.model.target_tract_dict[tt]] * scale_hazard
                for tt in target_tracts]
            all_haz_sum = np.sum(all_hazards)
            min_time = expon.rvs(scale=1.0/all_haz_sum)
            race_winner = target_tracts[
                np.random.choice(len(target_tracts), p=np.array(all_hazards)/all_haz_sum)]
        return race_winner, min_time

    def simulate(self,
            init_allele: Allele,
            node: CellLineageTree,
            scale_hazard_func = lambda x: 1,
            time_incr: float = 0.025):
        """
        @param init_allele: the initial state of the allele
        @param node: node to perform allele mutation from (this is the beginning node)
        @param scale_hazard_func: a function that takes in the current time in the tree
                        and returns how much to scale the cut rates
        @param time_incr: how much time to simulate allele mutation for

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

        Updates the allele in place. also returns the event descriptors

        @return left del len, right del len, insertion str
        """
        target1 = target_tract.min_target
        target2 = target_tract.max_target
        is_intertarg = target1 != target2

        left_long = int(target_tract.is_left_long)
        right_long = int(target_tract.is_right_long)

        # Keep running this until at least a trim or insertion occurs
        did_something = False
        while not did_something:
            do_insertion = random() > self.insert_zero_prob
            do_deletion = [
                random() > self.trim_zero_prob_dict[0,int(is_intertarg)],
                random() > self.trim_zero_prob_dict[1,int(is_intertarg)]]
            if is_intertarg or left_long or right_long:
                did_something = True
            else:
                did_something = bool(do_insertion + sum(do_deletion))

        # Determine insertion seq
        insert_len = self.insertion_distribution.rvs() if do_insertion else 0
        # Simple insertion mechanism for now
        insertion = ''.join(choice(list('acgt'), insert_len))

        # Determine deletion lengths
        left_del_len = self.left_del_dist[target1][left_long].rvs() if (do_deletion[0] or left_long) else 0
        right_del_len = self.right_del_dist[target2][right_long].rvs() if (do_deletion[1] or right_long) else 0

        allele.indel(
            target1,
            target2,
            left_del_len,
            right_del_len,
            insertion)
        return left_del_len, right_del_len, insertion
