import numpy as np
import scipy.stats

class SmoothBoxEnergy:
    def __init__(self, lo, hi, scale):
        self.lo = lo
        self.hi = hi
        self.scale = scale
        self.norm_coef = scipy.stats.halfcauchy.pdf(0, scale=scale)

    # energy density function (analog to PDF but unlike PDF integral over EDF is not one)
    def edf(self, x):
        return np.where(
            x < self.lo,
            scipy.stats.halfcauchy.pdf(self.lo - x, scale=self.scale) / self.norm_coef,
            np.where(
                x > self.hi,
                scipy.stats.halfcauchy.pdf(x, loc=self.hi, scale=self.scale) / self.norm_coef,
                1,
            )
        )    