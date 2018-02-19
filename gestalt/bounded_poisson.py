import numpy as np
from scipy.stats import poisson

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
        self.poisson_dist = poisson(poisson_param)

    def rvs(self, size=None):
        poiss_rv = self.min_val + self.poisson_dist.rvs(size=size)
        mask = poiss_rv <= self.max_val
        return np.multiply(poiss_rv, mask)

    def pmf(self, k: int):
        if k > self.max_val:
            # TODO: keep this here for simulations. remove later for real data
            raise ValueError("CUT TOO MUCH! %d, max = %d" % (k, self.max_val))
        elif k == 0:
            return self.poisson_dist.pmf(0) + 1 - self.poisson_dist.cdf(self.max_val)
        else:
            return self.poisson_dist.pmf(k)

    def __str__(self):
        return "bd_pois: %d, %d, %f" % (self.min_val, self.max_val, self.poisson_param)
