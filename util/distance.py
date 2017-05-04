import numpy as np

_eps = 1e-5

def euclidean_squared(X1, X2):
    """
    Efficient square 2D Euclidean distance computation between X1 and X2. Compatible with 
    data['X'] for Localization classes and functions.
    Note: Even identical X1, X2 do not output zero for numerical stability purposes, change _eps
    to alter this behavior

    X1:
        numpy array [n1,2] 
    X2:
        numpy array [n2,2] 

    output:
        numpy array [n1,n2]

    <math> 
        out[i,j] = |X1[i,:]-X2[j,:]|^2
    """

    X1sq = np.sum(np.square(X1),1)
    X2sq = np.sum(np.square(X2),1)
    r2 = -2.*np.dot(X1, X2.T) + X1sq[:,None] + X2sq[None,:]
    r2 = np.clip(r2, _eps, np.inf)
    return r2
