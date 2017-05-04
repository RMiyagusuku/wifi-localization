import numpy as np
import cPickle as pickle

def mesh(x1,x2):
    """
    Formated mesh of vectors x1, x2
    Output is numpy array [p,2], congruent with 'X' matrix used for positions in Localization
    functions and classes
    """
    x1v, x2v = np.meshgrid(x1,x2)
    x1v = np.reshape(x1v,(np.prod(x1v.shape),1))
    x2v = np.reshape(x2v,(np.prod(x2v.shape),1))
    XM  = np.concatenate((x1v,x2v),axis=1)
    return XM


    def save(var,filepath='var.p'):
        """
        Pickle var. 
        If no filepath is given, it is saved at <currentpath>/var.p
        """
        try: 
            import cPickle as pickle
        except ImportError:
            import pickle

        with open(filepath, 'wb') as f:
            pickle.dump(var,f,2)
        return True

    def load(filepath='last_model.p'):
        """
        Load a previously pickled var from filepath. 
        If no filepath is given, <currentpath>/var.p
        """
        try: 
            import cPickle as pickle
        except ImportError:
            import pickle

        with open(filepath, 'rb') as f:
            var = pickle.load(f)
        return var

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

