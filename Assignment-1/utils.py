from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    YTrue=np.array(y_true)
    YPred=np.array(y_pred)
    #print(YTrue)
    #print(YPred)
    return((YTrue - YPred) ** 2).mean(axis=None)
    #raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    #Real=numpy.array()
    tp=0
    tn=0
    fn=0
    fp=0
    for i in range(len(real_labels)):
        if(real_labels[i]==1 and predicted_labels[i]==1):
            tp+=1
        if (real_labels[i] == 0 and predicted_labels[i] == 0):
            tn+=1
        if (real_labels[i] == 1 and predicted_labels[i] == 0):
            fn+=1
        if (real_labels[i] == 0 and predicted_labels[i] == 1):
            fp+=1
    #print ("hello",tp,fp,fn)
    if tp+fp==0:
        precision=0
    else:
        precision=tp/(tp+fp)
    if(tp+fn==0):
        recall=0
    else:
        recall=tp/(tp+fn)
    #print(precision,recall)
    if (precision+recall)==0:
        return 0
    f1score=2*(precision*recall)/(precision+recall)

    return f1score
    #raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    Xpoly=np.array(features)
    #print("before",Xpoly)
    s=0
    for j in range(len(Xpoly)):
        Xtemp=Xpoly[j]
        Xpolytemp=Xtemp
        for i in range(2,k+1):
            #print(Xtemp)
            Xpower=np.power(Xtemp,i)
            Xpolytemp= np.hstack([Xpolytemp,Xpower])
            #print(Xpoly)
        if((s==0)):
            Xpolyvtemp=Xpolytemp
            s=1
        else:
            Xpolyvtemp=np.vstack([Xpolyvtemp,Xpolytemp])

    #print(Xpoly)
    Xpolyvtemp=np.delete(Xpolyvtemp,0,1)
    #print(Xpolyvtemp)
    Xpoly=np.hstack([Xpoly,Xpolyvtemp])
    #print(k,"after",Xpoly)
    return (Xpoly)
    #raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    X=np.array(point1)
    Y=np.array(point2)
    return (np.linalg.norm(X-Y))

    #raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    X=np.array(point1)
    Y=np.array(point2)
    return (np.dot(X,Y))
    #raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    X=np.array(point1)
    Y=np.array(point2)
    return -(np.exp(-(1/2)*np.power(np.linalg.norm(X-Y),2)))
    #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        self.norm=None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        X = np.array(features)
        if (not X.any()):
            return X.tolist()
        if(self.norm is None):
            self.norm=np.linalg.norm(X)
        Xbar = X / self.norm
        return Xbar.tolist()
        #raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
          self.mins=None
          self.maxs=None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        Xbar=np.array(features)
        #print (Xbar)
        (rows, cols) = Xbar.shape
        if((self.mins is None)):
            #print("One")
            self.mins=np.zeros(shape=(cols))
            self.maxs=np.zeros(shape=(cols))
            for j in range(cols):
                self.mins[j]=np.min(Xbar[:,j])
                self.maxs[j]=np.max(Xbar[:,j])

        result = np.copy(Xbar)
        for i in range(rows):
            for j in range(cols):
                result[i][j] = (Xbar[i, j] - self.mins[j]) / (self.maxs[j] - self.mins[j])
        #print ("Shape",Xbar.shape,result.shape)
        #print("result")
        return (result.tolist())


        #raise NotImplementedError
