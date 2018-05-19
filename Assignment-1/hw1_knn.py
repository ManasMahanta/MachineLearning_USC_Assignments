from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.X=numpy.array(features)
        self.Y=numpy.array(labels)
        #print (self.X)
        #print(self.Y)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        from collections import Counter
        Y=self.Y
        X=self.X
        K=self.k
        test=numpy.array(features)
        winners=[]
        for testindex in range(len(test)):
            test_instance=test[testindex]
            distances = []
            for index in range(len(X)):
                dist = self.distance_function(test_instance, X[index])
                distances.append((X[index], dist, Y[index]))
            distances.sort(key=lambda x: x[1])
            neighbors=distances[:K]
            class_counter = Counter()
            for neighbor in neighbors:
                class_counter[neighbor[2]] += 1
            winner=class_counter.most_common(1)[0][0]
            winners.append(winner)
        #print(winners)
        return winners
        #raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)