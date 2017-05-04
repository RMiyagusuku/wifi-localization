import numpy as np
import copy

def data_structure(din):
    """
    Script to verify if din is a dictionary compatible with Localization classes and functions.
    If not compatible, an error is raised. Else, a formated (if necessary) deep copy is returned 
    
    data:
        structure used for Localization classes and functions
        mandatory keys => 'X': numpy array [p,2]
                          'Y': numpy array [p,n] or [p,] - [p,n] should be formatted to [p,1]
                          'Var': numpy array [p,n] or [p,] - [p,n] should be formatted to [p,1]
                          
    out:
        formated deep copy of data
    """
        
    if not all(k in din for k in ('X','Y','Var')):  
        raise IOError       # If all keys are not present in dict, raise error

    assert din['X'].shape[1] == 2
    assert din['Y'].shape[0] == din['Y'].shape[0]
    assert din['Var'].shape[0] == din['Var'].shape[0]

    data = copy.deepcopy(din)

    if data['Y'].ndim == 1:
        data['Y'] = np.reshape(data['Y'],(data['Y'].shape[0],1))       #casting into [p,1]
        
    if data['Var'].ndim == 1:
        data['Var'] = np.reshape(data['Var'],(data['Var'].shape[0],1)) #casting into [p,1]

    return data

def angle(var):
    """
    Script to limit angle inputs between -pi to pi
    """
    return (var+np.pi)%(2*np.pi)-np.pi


