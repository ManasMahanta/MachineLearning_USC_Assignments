from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    unit_step = lambda x: -1 if x < 0 else 1

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]

            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        #print(np.array(features).shape,np.array(labels).shape)
        X=np.array(features)
        #print(X)
        Y=np.array(labels)
        w=np.zeros(len(X[0]))
        cur_iteration=0
        while True:
            done=True
            for i,x in enumerate(X):
                if (np.dot(X[i],w)*Y[i]) <= 0:
                    done=False
                    w = w + (X[i]*Y[i])/np.linalg.norm(X[i])

            if(done is True):
                break
            cur_iteration+=1
            if (cur_iteration >= self.max_iteration):
                self.w=w
                return False
        self.w=w

        return True


        #for m in range(self.max_iteration):
        #    for i,x in enumerate(X):
        #        if (np.dot(X[i], w)*Y[i]) <= 0:
        #                w = w + (X[i]*Y[i])/np.linalg.norm(X[i])
        #                #print (w)
        #self.w=w

        #print(m,w)

        #raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        unit_step = lambda x: 0 if x < 0 else 1
        #print (self.w,"Manas")
        X=np.array(features)
        result = np.dot(X,self.w)
        yhat=np.zeros(result.shape)
        #print (yhat)
        z=0
        for i in result:
            if i<0:
                yhat[z]=-1
                z+=1
            else:
                yhat[z]=1
                z+=1

        #yhat=-1 if result < 0 else 1
        #print (yhat)
        return yhat

        #print(result)
        #print("{}: {} -> {}".format(X[1:], result, unit_step(result)))
        #raise NotImplementedError

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    