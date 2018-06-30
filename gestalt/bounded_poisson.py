import numpy as np
from scipy.stats import poisson, randint

class BoundedPoisson:
    def __init__(self, min_val: int, max_val: int, poisson_param: float):
        raise NotImplementedError()

class ZeroInflatedBoundedPoisson(BoundedPoisson):
    """
    A poisson-like distribution that has a min and max value.
    If over maximum, then returns zero.
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
        poiss_raw = self.poisson_dist.rvs(size=size) + self.min_val
        poiss_bounded  = (
                poiss_raw * (poiss_raw <= self.max_val)
                + self.min_val * (poiss_raw > self.max_val))
        return poiss_bounded

    def pmf(self, k: int):
        if k > self.max_val or k < self.min_val:
            raise ValueError("Not in the support! %d, max = %d, min = %d" % (k, self.max_val, self.min_val))
        elif k == self.min_val:
            return self.poisson_dist.pmf(0) + 1 - self.poisson_dist.cdf(self.max_val - self.min_val)
        else:
            return self.poisson_dist.pmf(k - self.min_val)

    def __str__(self):
        return "zero inflated bounded pois: %d, %d, %f" % (self.min_val, self.max_val, self.poisson_param)

class PaddedBoundedPoisson(BoundedPoisson):
    """
    A poisson-like distribution that has a min and max value.
    If over the maximum, we pick a number in the support uniformly at random.
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
        self.uniform_dist = randint(min_val, max_val + 1)

    def rvs(self, size=None):
        poiss_raw = self.poisson_dist.rvs(size=size) + self.min_val
        poiss_bounded = (
                (poiss_raw <= self.max_val) * poiss_raw
                + (poiss_raw > self.max_val) * self.uniform_dist.rvs(size=size))
        return poiss_bounded

    def pmf(self, k: int):
        if k > self.max_val or k < self.min_val:
            raise ValueError("NOT in the support! %d, max = %d, min = %d" % (k, self.max_val, self.min_val))
        else:
            return (
                    self.poisson_dist.pmf(k - self.min_val)
                    + (1.0 - self.poisson_dist.cdf(self.max_val - self.min_val)) * self.uniform_dist.pmf(k))

    def __str__(self):
        return "padded bounded pois: %d, %d, %f" % (self.min_val, self.max_val, self.poisson_param)
