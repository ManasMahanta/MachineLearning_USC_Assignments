import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None
        from collections import namedtuple



    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k
            n_cluster=self.n_cluster
            max_iter=self.max_iter
            e=self.e
            k_means = KMeans(n_cluster, max_iter, e)
            self.means, _,_ = k_means.fit(x)
            #print(mu.shape)
            #self.means=np.random.random_sample((self.n_cluster, D))
            self.variances= np.array([np.eye(D)] * self.n_cluster)
            self.pi_k = np.array([1./self.n_cluster] * self.n_cluster)
            R = np.zeros((N, self.n_cluster))

            P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-x.shape[1]/2.) * np.exp(-.5 * np.einsum('ij, ij -> i',x - mu, np.dot(np.linalg.inv(s) , (x - mu).T).T ) )
            curr_iter=1
            log_likelihoods = []
            while (len(log_likelihoods) < self.max_iter):
                #print(len(log_likelihoods))
                import sys
                for k in range(self.n_cluster):
                    #R[:, k] = self.pi_k[k] * P(self.means[k], self.variances[k])
                    #print(np.multiply(np.eye(self.variances[k].shape[1]),0.001))
                    if np.linalg.matrix_rank(self.variances[k])==self.variances[k].shape[1]:
                        R[:, k] = self.pi_k[k] * P(self.means[k], self.variances[k])
                        #print ("true")
                    else:
                        S=np.add(self.variances[k],(np.multiply(np.eye(self.variances[k].shape[1]),0.001)))
                        if np.linalg.matrix_rank(S)==S.shape[1]:
                            R[:, k] = self.pi_k[k] * P(self.means[k], S)
                            #print ("true1")
                        else:
                            #print ("true2")
                            S2=np.add(S,(np.multiply(np.eye(S.shape[1]),0.001)))
                            R[:, k] = self.pi_k[k] * P(self.means[k], S2)

                #if(np.sum(R, axis = 1).all()==0):
                #    print ("there1")

                log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
                log_likelihoods.append(log_likelihood)

                R = (R.T / np.sum(R, axis = 1)).T
                N_ks = np.sum(R, axis = 0)






                #print(N_ks)

                for k in range(self.n_cluster):
                    self.means[k] = 1. / N_ks[k] * np.sum(R[:, k] * x.T, axis = 1).T
                    x_mu = np.matrix(x - self.means[k])
                    self.variances[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                    self.pi_k[k] = 1. / N * N_ks[k]

                #print(len(log_likelihoods),self.means)
                #print(np.abs(log_likelihood - log_likelihoods[-2]))


                if len(log_likelihoods) < 2 : continue
                #print(log_likelihoods[-1])
                if np.abs(log_likelihood - log_likelihoods[-2]) < self.e:
                    #print(len(np.unique(self.means, axis=0)))
                    break

            #print("loglikelihood",log_likelihoods[-1])
            #self.means=np.array(mu)
            #self.variances=np.array(Sigma)
            #self.pi_k=np.array(w)


            #print(log_likelihoods[-1])
            return (len(log_likelihoods))


            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            #self.variances=[np.eye(D)]*self.n_cluster
            #self.means=x[np.random.choice(np.arange(N),self.n_clusters),:]
            #self.pi_k=[1/K]*K

            #mu = [np.random.uniform(0,1,self.n_cluster)]
            self.means=np.random.random_sample((self.n_cluster, D))
            self.variances= np.array([np.eye(D)] * self.n_cluster)
            self.pi_k = np.array([1./self.n_cluster] * self.n_cluster)
            R = np.zeros((N, self.n_cluster))

            P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-x.shape[1]/2.) * np.exp(-.5 * np.einsum('ij, ij -> i',x - mu, np.dot(np.linalg.inv(s) , (x - mu).T).T ) )
            curr_iter=1
            log_likelihoods = []
            while (len(log_likelihoods) < self.max_iter):
                #print(len(log_likelihoods))
                import sys
                for k in range(self.n_cluster):
                    #R[:, k] = self.pi_k[k] * P(self.means[k], self.variances[k])
                    if np.linalg.matrix_rank(self.variances[k])==self.variances[k].shape[1]:
                        R[:, k] = self.pi_k[k] * P(self.means[k], self.variances[k])
                        #print ("true")
                    else:
                        S=np.add(self.variances[k],(np.multiply(np.eye(self.variances[k].shape[1]),0.001)))
                        if np.linalg.matrix_rank(S)==S.shape[1]:
                            R[:, k] = self.pi_k[k] * P(self.means[k], S)
                            #print ("true1")
                        else:
                            #print ("true2")
                            S2=np.add(S,(np.multiply(np.eye(S.shape[1]),0.001)))
                            R[:, k] = self.pi_k[k] * P(self.means[k], S2)


                log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
                log_likelihoods.append(log_likelihood)

                R = (R.T / np.sum(R, axis = 1)).T
                N_ks = np.sum(R, axis = 0)

                #print()

                for k in range(self.n_cluster):
                    self.means[k] = 1. / N_ks[k] * np.sum(R[:, k] * x.T, axis = 1).T
                    x_mu = np.matrix(x - self.means[k])
                    self.variances[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                    self.pi_k[k] = 1. / N * N_ks[k]




                if len(log_likelihoods) < 2 : continue
                #print(log_likelihoods[-1])
                if np.abs(log_likelihood - log_likelihoods[-2]) < self.e:
                    #print(len(np.unique(self.means, axis=0)))
                    break

            #print("loglikelihood",log_likelihoods[-1])

            #self.variances=np.array(Sigma)
            #self.pi_k=np.array(w)

            #print(log_likelihoods[-1])

            return (len(log_likelihoods))

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement fit function (filename: gmm.py)')
        # DONOT MODIFY CODE BELOW THIS LINE



    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples


        sample=[]
        for i in np.random.choice(self.n_cluster, size=N, p=self.pi_k):
            mu=self.means[i]
            sigma=self.variances[i]
            sample.append(np.random.multivariate_normal(mu,sigma))

        return(np.array(sample))
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N,D=x.shape
        Sigma=self.variances
        mu=self.means
        w=self.pi_k
        R = np.zeros((N, self.n_cluster))
        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-x.shape[1]/2.) * np.exp(-.5 * np.einsum('ij, ij -> i',x - mu, np.dot(np.linalg.inv(s) , (x - mu).T).T ) )
        for k in range(self.n_cluster):
            if np.linalg.matrix_rank(Sigma[k])==Sigma[k].shape[1]:
                R[:, k] = w[k] * P(mu[k], Sigma[k])
                #print ("true")
            else:
                S=np.add(Sigma[k],(np.multiply(np.eye(Sigma[k].shape[1]),0.001)))
                if np.linalg.matrix_rank(S)==S.shape[1]:
                    R[:, k] = w[k] * P(mu[k], S)
                    #print ("true1")
                else:
                    #print ("true2")
                    S2=np.add(S,(np.multiply(np.eye(S.shape[1]),0.001)))
                    R[:, k] = w[k] * P(mu[k], S2)
        #if(np.sum(R, axis = 1).all()==0):
        #    print ("there2")
        log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))

        '''
        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-x.shape[1]/2.) * np.exp(-.5 * np.einsum('ij, ij -> i',x - mu, np.dot(np.linalg.inv(s) , (x - mu).T).T ) )
        R = np.zeros((N, self.n_cluster))
        for k in range(self.n_cluster):
            R[:, k] = self.pi_k[k] * P(self.means[k],self.variances[k])
        log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))'''
        #print(log_likelihood)

        return float(log_likelihood)


        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE