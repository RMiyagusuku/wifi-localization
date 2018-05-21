from __future__ import print_function

import numpy as np
import scipy
import sys
#import sensorfusion 

def eprint(*args, **kwargs):                    #prints errors/warnings to stderr
    print(*args, file=sys.stderr, **kwargs)


class Likelihood(object):
    """
    Base class for likelihoods. Each inherited classs needs to implement jointpdf
    """
    def __init__(self,**kwargs):
        super(Likelihood,self).__init__()
        self.name = 'likelihood' 

    def likelihood(self,model,Xn,Yn):
        """
        Computes the likelihood for all models for measurements Yn
        to have been taken at positions Xn.

        """
        raise NotImplementedError

    def jointpdf(self,model,Xn,Yn):
        """
        Computes de joint likelihood of all models for measurements Yn
        to have been taken at positions Xn.
        """
        raise NotImplementedError 

class Gaussian(Likelihood):
    """
    Gaussian likelihood default jointpdf uses the geometric mean likelihood of all models.
    
    kwargs  
        deltax: see prob()
        confidence_zero: 
        pzero: 
    """
    def __init__(self,**kwargs):
        super(Gaussian,self).__init__()
        self.name            = 'PoE Gaussian likelihood'
        self.deltax          = kwargs.pop('deltax',.025)
        self.confidence_zero = kwargs.pop('confidence_zero',0.)
        self.pzero           = kwargs.pop('pzero',.05)

    def prob_sensormodel(self,x,mu,var,**kwargs):
        """
        Calculates the probability of each individual access point by integrating the density
        function of pzero*deltaz + (1-pzero)phi(mu-x,var)[from x-deltax to x+deltax].
        with deltaz == 1 iff x=0, 0 otherwise
        Parameters
            x: float or ndarray
                point which probability is calculated
            mu, var: floats or ndarrays
                mean and variance of the gausian density
            pzero: float, optional
                mass point distribution at zero            
            deltax: float, optional 
                variation around x where the integration is performed - default .15

        <math>    
            cdf(x) = 0.5*(1+erf((x-mu)/sqrt(2var)))
            prob = cdf(x+deltax)-cdf(x-deltax)
        """
        pzero  = kwargs.get('pzero',self.pzero)                     #0.05 
        deltax = kwargs.get('deltax',self.deltax)                   #0.025

        xplus  = (x+deltax-mu)/(var*2)**0.5
        xless  = (x-deltax-mu)/(var*2)**0.5
        gpx    = 0.5*(scipy.special.erf(xplus)-scipy.special.erf(xless))
        zpx    = np.zeros_like(x)         # Probability if the measurement is zero
        zpx[np.where(x==0)] = pzero
        px     = zpx + (1-pzero)*gpx

        return np.clip(px,1e-15,np.inf)

    def likelihood(self,model,Xn,Yn,**kwargs):
        """
        Parameters:
            model: SensorModel
                SensorModel class that generates the predictions
            Xn: ndarray [ntp,2]
                ntp test points
            Yn: ndarray [1,nap] or [nap,]
                signal measurement vectors from all nap access points
           
        Returns
            pX: ndarray [ntp,nap]
                likelihood for all ntp testpoints given each access point model

        """
        ntp  = Xn.shape[0]  #number of X test points
        nap  = model.nap    #number of access points

        measurement = Yn.copy().reshape(1,nap) #Yn becomes [1,nap]

        #check inputs
        assert measurement.shape[1]==nap
        assert Xn.shape[1] == 2

        #model prediction
        ypredict, varpredict = model.predict(Xn)       
        
        #compute likelihood for all testpoints and access points    
        #if varpredict.shape[1] == 1:
        var  = varpredict.copy()
            
        xs   = measurement
        mus  = ypredict
        #if varpredict.shape[1] > 1:
        #    var  = varpredict#[:,i:i+1] #error fix
    
        pX   = self.prob_sensormodel(xs,mus,var)

        return pX


    def jointpdf(self,model,Xn,Yn,**kwargs):
        """
        model:
            mapping class that generates the predictions for Xn
        Xn:
            np test points as a numpy array [np, 2]
        Yn: 
            np measurement points of all nap wireless measurements as a numpy array [np, nap]
        """
        
        confidence_zero = kwargs.pop('confidence_zero',self.confidence_zero)

        pX = self.likelihood(model,Xn,Yn,**kwargs)
        log_pX = np.log(pX)

        alpha = np.ones(model.nap)
        #reduce confidence of models where signal is zero 
        index = np.where(Yn==0)
        alpha[index[0]] *= confidence_zero
        #normalize to one
        alpha = alpha/np.sum(alpha) 
        px = np.dot(log_pX,alpha) #log_px*alpha = prod(pX[i]*alpha[i])
            
        return np.exp(px)

