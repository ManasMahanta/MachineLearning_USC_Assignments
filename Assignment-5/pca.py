import numpy as np

def pca(X = np.array([]), no_dims = 50):
    """
    Runs PCA on the N x D array X in order to reduce its dimensionality to 
     no_dims dimensions.
    Inputs:
    - X: A matrix with shape N x D where N is the number of examples and D is 
         the dimensionality of original data.
    - no_dims: A scalar indicates the output dimension of examples after 
         performing PCA.
    Returns:
    - Y: A matrix of reduced size with shape N x no_dims where N is the number
         of examples  and no_dims is the dimensionality of output examples. 
         no_dims should be smaller than D, which is the dimensionality of 
         original examples.
    - M: A matrix of eigenvectors with shape D x no_dims where D is the 
         dimensionality of the original data
    """
    Y = np.array([])
    M = np.array([])

    #print(X)
    N,D=X.shape

    mean_vec=np.mean(X,axis=0)
    cov_mat=(X-mean_vec).T.dot((X-mean_vec))/(X.shape[0]-1)
    eig_vals,eig_vecs=np.linalg.eig(cov_mat)
    #for ev in eig_vec:
    #    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    #print('Everything ok!')

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    #print('Eigenvalues in descending order:')
    #for i in eig_pairs:
    #    print(i[0])
    #print (eig_vals,eig_vec)
    #print(np.unique(cov_mat))
    #print('Covariance matrix \n%s' %cov_mat)
    M = np.hstack((eig_pairs[0][1].reshape(D,1),
                      eig_pairs[1][1].reshape(D,1)))

    for k in range(2,no_dims):
        M=np.hstack((M,eig_pairs[k][1].reshape(D,1)))

    """TODO: write your code here"""

    Y = X.dot(M)

    #print(Y.shape,M.shape)
    
    return Y, M

def decompress(Y = np.array([]), M = np.array([])):
    """
    Returns compressed data to initial shape, hence decompresses it.
    Inputs:
    - Y: A matrix of reduced size with shape N x no_dims where N is the number
         of examples  and no_dims is the dimensionality of output examples. 
         no_dims should be smaller than D, which is the dimensionality of 
         original examples.
    - M: A matrix of eigenvectors with shape D x no_dims where D is the 
         dimensionality of the original data
    Returns:
    - X_hat: Reconstructed matrix with shape N x D where N is the number of 
         examples and D is the dimensionality of each example before 
         compression.
    """
    X_hat = np.array([])

    """TODO: write your code here"""
    X_hat=Y.dot(M.T)
    
    return X_hat

def reconstruction_error(orig = np.array([]), decompressed = np.array([])):
    """
    Computes reconstruction error (pixel-wise mean squared error) for original
     image and reconstructed image
    Inputs:
    - orig: An array of size 1xD, original flattened image.
    - decompressed: An array of size 1xD, decompressed version of the image
    """
    error = 0
    error = np.real(np.mean((orig - decompressed)**2))

    """TODO: write your code here"""
    
    return error

def load_data(dataset='mnist_subset.json'):
    # This function reads the MNIST data
    import json


    with open(dataset, 'r') as f:
        data_set = json.load(f)
    mnist = np.vstack((np.asarray(data_set['train'][0]), 
                    np.asarray(data_set['valid'][0]), 
                    np.asarray(data_set['test'][0])))
    return mnist

if __name__ == '__main__':
    
    import argparse
    import sys


    mnist = load_data()
    #print(mnist)
    compression_rates = [2, 10, 50, 100, 250, 500]
    with open('pca_output.txt', 'w') as f:
        for cr in compression_rates:
            Y, M = pca(mnist - np.mean(mnist, axis=0), cr)
            
            decompressed_mnist = decompress(Y, M)
            decompressed_mnist += np.mean(mnist, axis=0)
            
            total_error = 0.
            for mi, di in zip(mnist, decompressed_mnist):
                error = reconstruction_error(mi, di)
                f.write(str(error))
                f.write('\n')
                total_error += error
            print('Total reconstruction error after compression with %d principal '\
                'components is %f' % (cr, total_error))



