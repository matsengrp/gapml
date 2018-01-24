import numpy as np
from scipy.misc import factorial

class BoundedPoisson:
    """
    A poisson-like distribution that has a min and max value
    and is normalized appropriately!
    """
    def __init__(self, min_val: int, max_val: int, poisson_param: float):
        """
        @param min_val: inclusive
        @param max_val: inclusive
        """
        self.min_val = min_val
        self.max_val = max_val
        self.poisson_param = poisson_param
        self.all_pmf = self._all_pmf(min_val, max_val, poisson_param)

    def rvs(self, size=None):
        return np.random.choice(np.arange(self.min_val, self.max_val + 1), p=self.all_pmf, size=size)

    def pmf(self, k: int):
        if k > self.max_val:
            # TODO: keep this here for simulations. remove later for real data
            raise ValueError("CUT TOO MUCH! %d, max = %d" % (k, self.max_val))
        return self.all_pmf[k - self.min_val]

    def __str__(self):
        return "bd_pois: %d, %d, %f" % (self.min_val, self.max_val, self.poisson_param)

    @staticmethod
    def _all_pmf(min_val: int, max_val: int, poisson_param: float):
        p_vals_unstd = [
            BoundedPoisson._get_pmf_unstd(i, min_val, max_val, poisson_param)
            for i in range(min_val, max_val + 1)]
        return p_vals_unstd/sum(p_vals_unstd)

    @staticmethod
    def _get_pmf_unstd(val: int, min_val: int, max_val: int, poisson_param: float):
        """
        @return un-normalized prob of observing `val` for a bounded poisson
        """
        if val > max_val:
            return 0
        else:
            return float(np.power(poisson_param, val - min_val))/factorial(val - min_val)
