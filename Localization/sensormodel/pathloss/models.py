import numpy as np
from util.distance import euclidean_squared
from .classes import PathLoss

class FLog(PathLoss):
    """
    Path loss function with power loss in dB logarithmic in distance    
    """
    def __init__(self,data,**kwargs):
        super(FLog,self).__init__(data,**kwargs)
        self.name = 'Log Function'
        if self.debug:
            print 'class FLog init works'
        
    
    def mean_val(self,X,params=None,bounded=True):# params=None,i=0):
        """
        Compute the estimated value of the function at points X - see .classes.PathLossFun

        <math>
            Pr = P0 - k*log10(|Xap-X|)
        """

        if params is None:
            params = self.params

        if params.ndim == 1:
            params = np.reshape(params,(1,params.shape[0]))
    

        XAP = params[:,0:2]
        k  = 0.8 #0.5 + 0.25*np.cos(params[:,2])
        P0 = 0.9 + .1*np.cos(params[:,3])    

        r = euclidean_squared(X,XAP)**0.5
        out = P0 - k*np.log10(r)

        if bounded:
            return np.clip(out,0.,P0)
        else:
            return np.clip(out,-np.inf,P0) 


    def params_init(self):
        """
        Generate suitable initial random parameters - see .classes.PathLossFun
            Considers ap source X position as the one with the highest Y value, sd is set
            to .1*max_X
            k and P0 related parameters are the angle for a cos function, so the optimization
            is bounded to +- cos amplitude, initialization is set to 0 with sd of 1
                
            returns a numpy array [p,4]

        <math>
            ap = argmax_X(Y) +- .1*max(X)
            k  = 1.5 + cos(0 +- 1)
            P0 = 0.9 + .1*cos(0 +- 1)
        """
        max_arg_y = np.argmax(self.data['Y'],axis=0)
        ap0 = self.data['X'][max_arg_y]
        ap0_sd = 0.1*np.max(np.abs(self.data['X']))
        temp_ones = np.ones_like(ap0[:,0])
        return np.random.normal(
               loc  = (ap0[:,0],ap0[:,1],0*temp_ones,0*temp_ones), 
               scale= (ap0_sd*temp_ones,ap0_sd*temp_ones,temp_ones,temp_ones)).T


        

