from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        X=numpy.array(features)
        #print(((X)))
        #print(numpy.ones((len(X),1)))
        Xbar=numpy.hstack([numpy.ones((len(X),1)),X])
        Y=numpy.array(values)
        W=numpy.array(numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(Xbar.T, Xbar)), Xbar.T), Y))
        #print(Xbar)
        #print(Xbar.shape)
        #print(Y)
        #print(Y.shape)
        self.W=W
        #print(W)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        W=self.W
        X=numpy.array(features)
        Xbar=numpy.hstack([numpy.ones((len(X),1)),X])
        #print(W.T)
        #print(W)
        #print(Xbar.shape)
        return(numpy.dot(Xbar,W))
        #raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.W
        #raise NotImplementedError


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        X=numpy.array(features)
        Xbar=numpy.hstack([numpy.ones((len(X),1)),X])
        Y=numpy.array(values)
        W= numpy.array(numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(Xbar.T, Xbar)+self.alpha*numpy.identity(Xbar.shape[1])), Xbar.T), Y))
        self.W=W
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        W=self.W
        X=numpy.array(features)
        Xbar=numpy.hstack([numpy.ones((len(X),1)),X])
        return(numpy.dot(Xbar,W))

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.W
        #raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
