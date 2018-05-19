import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape
        K=self.n_cluster

        centroids = x[np.random.choice(np.arange(len(x)), K), :]
        current_iter=0
        #C_old=np.zeros((N,self.n_cluster))

        J_old=999999999999
        for i in range(self.max_iter):
            # Cluster Assignment step
            current_iter+=1
            C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in x])
            centroids = [x[C == k].mean(axis = 0) for k in range(K)]

            R=np.zeros((N,K))
            R[np.arange(N),C]=1
            #print (R)

            # Move centroids step
            #print()
            #print(self.e)
            #print(max(3.592567987660382e-05,self.e))

            J=(1/N)*(np.sum(np.multiply(R,[[np.dot(x_i-y_k, x_i-y_k) for y_k in centroids] for x_i in x])))
            #print(current_iter,np.abs(J_old-J))
            if np.abs(J_old-J)<=self.e:
                number_of_updates=current_iter
                #print("Yess")
                break

            J_old=J

            #print(current_iter)
        #print(N,D)
        #print(x)
        return (np.array(centroids),C,current_iter)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeans class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        n_cluster=self.n_cluster
        max_iter=self.max_iter
        e=self.e

        k_means = KMeans(n_cluster,max_iter,e)
        centroids, membership, _ = k_means.fit(x)

        R=np.zeros((N,n_cluster))
        R[np.arange(N),membership]=1

        #labels= np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for c in centroids])
        centroid_labels=np.zeros(self.n_cluster)
        for k in range(n_cluster):
            vote=np.zeros(len(set(y)))
            for i in range(N):
                if(R[i][k]==1):
                    vote[y[i]]+=1
            centroid_labels[k]=np.argmax(vote)

        #print(centroid_labels)
        #print(N)




        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in self.centroids]) for x_i in x])
        #print(C.shape,x.shape)
        pred=np.zeros((N,))
        for i in range(C.shape[0]):
            pred[i]=self.centroid_labels[C[i]]

        #print(pred)
        return pred
        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE
