import numpy as np
from scipy.stats import poisson, randint, nbinom
from common import sigmoid

class BoundedPoisson:
    def __init__(self, min_val: int, max_val: int, poisson_param: float):
        raise NotImplementedError()

class ShiftedPoisson(BoundedPoisson):
    """
    A poisson-like distribution that has a min value.
    """
    def __init__(self, min_val: int, poisson_param: float):
        """
        @param min_val: inclusive
        @param max_val: inclusive
        """
        self.min_val = min_val
        self.poisson_param = poisson_param
        self.poisson_dist = poisson(poisson_param)

    def rvs(self, size=None):
        return self.poisson_dist.rvs(size=size) + self.min_val

    def pmf(self, k: int):
        if k < self.min_val:
            raise ValueError("Not in the support! %d, min = %d" % (k, self.min_val))
        else:
            return self.poisson_dist.pmf(k - self.min_val)

    def __str__(self):
        return "shifted pois: %d, %f" % (self.min_val, self.poisson_param)

class ConditionalBoundedPoisson(BoundedPoisson):
    """
    A poisson-like distribution that has a min and max value, conditional within the range
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
        if size is None:
            rv = self.poisson_dist.rvs() + self.min_val
            while rv > self.max_val:
                rv = self.poisson_dist.rvs() + self.min_val
            return rv

        raw_rv = self.poisson_dist.rvs(size=size) + self.min_val
        raw_rv = raw_rv[raw_rv <= self.max_val]
        while raw_rv.size < size:
            new_rv = self.poisson_dist.rvs(size=size) + self.min_val
            new_rv = new_rv[new_rv <= self.max_val]
            raw_rv = np.concatenate(raw_rv, new_rv)
            print("TRY sampling bounded poiss again")

        rv_bounded = raw_rv[:size]
        return rv_bounded

    def pmf(self, k: int):
        if k > self.max_val or k < self.min_val:
            raise ValueError("NOT in the support! %d, max = %d, min = %d" % (k, self.max_val, self.min_val))
        else:
            return self.poisson_dist.pmf(k - self.min_val)/self.poisson_dist.cdf(self.max_val - self.min_val)

    def __str__(self):
        return "conditional bounded pois: %d, %d, %f" % (self.min_val, self.max_val, self.poisson_param)


class BoundedNegativeBinomial:
    def __init__(self, min_val: int, max_val: int, m: int, logit: float):
        raise NotImplementedError()

class ConditionalBoundedNegativeBinomial(BoundedNegativeBinomial):
    """
    A neg-binom like distribution that has a min and max value, conditional within the range
    """
    def __init__(self, min_val: int, max_val: int, m: int, logit: float):
        """
        @param min_val: inclusive
        @param max_val: inclusive
        """
        self.min_val = min_val
        self.max_val = max_val
        self.m = m
        self.prob = sigmoid(logit)
        self.nbinom_dist = nbinom(self.m, 1 - self.prob)

    def rvs(self, size=None):
        if size is None:
            rv = self.nbinom_dist.rvs() + self.min_val
            while rv > self.max_val:
                rv = self.nbinom_dist.rvs() + self.min_val
            return rv

        raw_rv = self.nbinom_dist.rvs(size=size) + self.min_val
        raw_rv = raw_rv[raw_rv <= self.max_val]
        while raw_rv.size < size:
            new_rv = self.nbinom_dist.rvs(size=size) + self.min_val
            new_rv = new_rv[new_rv <= self.max_val]
            raw_rv = np.concatenate(raw_rv, new_rv)

        rv_bounded = raw_rv[:size]
        return rv_bounded

    def pmf(self, k: int):
        if k > self.max_val or k < self.min_val:
            raise ValueError("NOT in the support! %d, max = %d, min = %d" % (k, self.max_val, self.min_val))
        else:
            return self.nbinom_dist.pmf(k - self.min_val)/self.nbinom_dist.cdf(self.max_val - self.min_val)

    def __str__(self):
        return "cond bounded nbinom: %d, %d, %d, %f" % (self.min_val, self.max_val, self.m, self.prob)

class ShiftedNegativeBinomial(BoundedNegativeBinomial):
    """
    A neg-binom like distribution that has a min and max value.
    If over the maximum, we pick a number in the support uniformly at random.
    """
    def __init__(self, min_val: int, m: int, logit: float):
        """
        @param min_val: inclusive
        @param max_val: inclusive
        """
        self.min_val = min_val
        self.m = m
        self.prob = sigmoid(logit)
        self.nbinom_dist = nbinom(self.m, 1 - self.prob)

    def rvs(self, size=None):
        return self.nbinom_dist.rvs(size=size) + self.min_val

    def pmf(self, k: int):
        if k < self.min_val:
            raise ValueError("NOT in the support! %d, min = %d" % (k, self.min_val))
        else:
            return self.nbinom_dist.pmf(k - self.min_val)

    def __str__(self):
        return "cond bounded nbinom: %d, %d, %f" % (self.min_val, self.m, self.prob)
