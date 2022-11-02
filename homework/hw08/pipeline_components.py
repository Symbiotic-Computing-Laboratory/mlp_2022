"""
Collection of class objects to utilize as Pipeline Components for data
pre-processing


CS 5703 Machine Learning Practice
Andrew H. Fagg (andrewhfagg@gmail.com)
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Pipeline component: select subsets of attributes
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        return X[self.attribs]

# Pipeline component: drop all rows that contain invalid values
class DataSampleDropper(BaseEstimator, TransformerMixin):
    def __init__(self, how='any'):
        self.how = how
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        return X.dropna(how=self.how)

# Pipeline component: fills NaNs
class InterpolationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='linear'):
        self.method = method
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        '''
        PARAMS:
            X: is a DataFrame
        RETURNS: a DataFrame without NaNs
        '''
        # Interpolate holes within the data
        Xout = X.interpolate(method=self.method)
        # Fill in the NaNs on the edges of the data
        Xout = Xout.fillna(method='ffill')
        Xout = Xout.fillna(method='bfill')
        return Xout

# Pipeline component: filter data with Gaussian kernel
def computeweights(length=3, sig=1):
    '''
    Computes the weights for a Gaussian filter kernel
    PARAMS:
        length: the number of terms in the filter kernel
        sig: the standard deviation (i.e. the scale) of the Gaussian
    RETURNS: a list of filter weights for the Gaussian kernel
    '''
    x = np.linspace(-2.5, 2.5, length)
    kernel = stats.norm.pdf(x, scale=sig)
    return kernel / kernel.sum()

# Pipeline component: filter data with Gaussian kernel
class GaussianFilter(BaseEstimator, TransformerMixin):
    def __init__(self, attribs=None, kernelsize=3, sig=1):
        self.attribs = attribs
        self.kernelsize = kernelsize
        self.sig = sig
        self.weights = computeweights(length=kernelsize, sig=sig)
        print("KERNEL WEIGHTS", self.weights)
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        '''
        PARAMS:
            X: is a DataFrame
        RETURNS: a DataFrame with the smoothed signals
        '''
        w = self.weights
        ks = self.kernelsize
        Xout = X.copy()
        if self.attribs == None:
            self.attribs = Xout.columns
        
        for attrib in self.attribs:
            vals = Xout[attrib].values
            # Amount to pad each end
            npad = int(ks / 2)
            # Total padding
            tpad = ks - 1
            # Pad each end
            frontpad = [vals[0]] * npad
            backpad = [vals[-1]] * npad
            vals = np.concatenate((frontpad, vals, backpad))
            # Apply filter
            avg = w[6] * vals[6:]
            for i in range(tpad): avg += w[i] * vals[i:(-tpad+i)]
            Xout[attrib] = pd.Series(avg)   
        return Xout

# Pipeline component: Compute derivatives
class ComputeDerivative(BaseEstimator, TransformerMixin):
    def __init__(self, attribs, dt=1.0, prefix='d_'):
        self.attribs = attribs
        self.dt = dt
        self.prefix = prefix
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        # Compute derivatives
        Xout = X.copy()
        for field in self.attribs:
            # Extract the values for this field
            values = Xout[field].values
            # Compute the difference between subsequent values
            diff = values[1:] - values[0:-1]
            # Bring the length to be the same as original data
            np.append(diff, 0)
            # Name of the new field
            name = self.prefix + field
            Xout[name] = pd.Series(diff / self.dt)
        return Xout
