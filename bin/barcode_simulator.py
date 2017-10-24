from numpy import ndarray
import numpy as np
from scipy.stats import expon, poisson
from numpy.random import choice

from barcode import Barcode


class BarcodeSimulator:
    """
    

    Assumes each barcode can have at most two cuts.
    """
    def __init__(self, target_lambdas: ndarray, repair_rates: ndarray, left_del_lambda: float, right_del_lambda: float, insertion_lambda: float):
        self.target_lambdas = target_lambdas
        self.repair_rates = repair_rates
        self.left_del_lambda = left_del_lambda
        self.right_del_lambda = right_del_lambda
        self.insertion_lambda = insertion_lambda

    def simulate(self, init_barcode: Barcode, time: float):
        barcode = Barcode(init_barcode.barcode, init_barcode.unedited_barcode, init_barcode.cut_sites)
        time_remain = time
        while time_remain > 0:
            repair_time = expon.rvs(scale=1.0/self.repair_rates[len(barcode.needs_repair) - 1]) if len(barcode.needs_repair) else np.inf
            if len(barcode.needs_repair) >= 2:
                # No more than two cuts allowed in the barcode
                event_times = [repair_time]
                target_cut_times = []
            else:
                # If barcode has <=1 cuts, then barcode may get cut
                active_targets = barcode.get_active_targets()
                target_cut_times = [expon.rvs(scale=1.0/self.target_lambdas[i]) for i in active_targets]
                event_times = target_cut_times + [repair_time]

            race_winner = np.argmin(event_times)
            event_time = np.min(event_times)
            time_remain = max(time_remain - event_time, 0)

            if time_remain == 0:
                # Time is up, so nothing happened
                time_remain = 0
            elif race_winner < len(target_cut_times):
                # One of the targets got cut
                barcode.cut(race_winner)
            else:
                target1 = min(barcode.needs_repair)
                target2 = max(barcode.needs_repair)
                left_del_len = poisson.rvs(self.left_del_lambda)
                right_del_len = poisson.rvs(self.right_del_lambda)
                insertion_length = poisson.rvs(self.insertion_lambda)
                # TODO: make this more realistic. right now just random DNA inserted
                insertion = ''.join(choice(list('acgt'), insertion_length))
                barcode.indel(target1, target2, left_del_len, right_del_len, insertion)
        return barcode


