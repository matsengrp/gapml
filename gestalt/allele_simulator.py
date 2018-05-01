from numpy import ndarray
import numpy as np

from allele import Allele


class AlleleSimulator:
    """
    Simulate how a allele gets cut/repaired.
    Subclass this if you want to experiment with other models of the allele cut/repair process.
    """
    def simulate(self, init_allele: Allele, time: float):
        """
        @param init_allele: the initial state of the allele
        @param time: the amount of time to simulate the allele modification process

        @return allele after the simulation procedure
                    does not modify the allele that got passed in :)
        """
        raise NotImplementedError()

    def get_root(self):
        """
        @return the root's AlleleList
        """
        raise NotImplementedError()
