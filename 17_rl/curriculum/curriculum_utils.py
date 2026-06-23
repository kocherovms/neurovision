import numpy as np
import scipy.stats

class SmoothBoxEnergy:
    def __init__(self, lo, hi, scale, min_right=0):
        self.lo = lo
        self.hi = hi
        self.scale = scale
        self.norm_coef = scipy.stats.halfcauchy.pdf(0, scale=scale)
        self.min_right = min_right

    # energy density function (analog to PDF but unlike PDF integral over EDF is not one)
    def edf(self, x):
        y = np.where(
            x < self.lo,
            scipy.stats.halfcauchy.pdf(self.lo - x, scale=self.scale) / self.norm_coef,
            np.where(
                x > self.hi,
                scipy.stats.halfcauchy.pdf(x, loc=self.hi, scale=self.scale) / self.norm_coef,
                1,
            )
        )

        if self.min_right > 0:
            y = np.where(x > self.hi, np.where(y > self.min_right, y, self.min_right), y)

        return y