from __future__ import print_function

import numpy as np
import scipy
import sys

def eprint(*args, **kwargs):                    #prints errors/warnings to stderr
    print(*args, file=sys.stderr, **kwargs)


class Likelihood(object):
    """
    Base class for likelihoods. Each inherited classs needs to implement jointpdf
    """
    def __init__(self,**kwargs):
        super(Likelihood,self).__init__()

    def jointpdf(self,Xn,Yn,Varn):
        """
        Computes de loglikelihood of measurements Yn with variance Varn to have been taken 
        at positions Xn.
        """
        raise NotImplementedError 

class Gaussian(Likelihood):
    """
    Gaussian likelihood of all accespoints calculated as the geometric mean of each one.
    
    kwargs  
        deltax: see prob()
        confidence_zero: 
        pzero: 
    """
    def __init__(self,**kwargs):
        super(Gaussian,self).__init__()
        self.deltax             = kwargs.pop('deltax',.025)
        self.confidence_zero    = kwargs.pop('confidence_zero',.3)
        self.pzero              = kwargs.pop('pzero',.05)
        #TODO:Change to 1
        self.falpha             = kwargs.pop('falpha',1.)

    def prob(self,x,mu,var,**kwargs):
        """
        Calculates the probability of each individual access point by integrating the density
        function of N(mu,var) from x-deltax to x+deltax.

        x: 
            point which probability is calculated
        mu, var:
            mean and variance of the gausian density
        kwargs    
            deltax: variation around x where the integration is performed - default .15

        <math>    
            cdf(x) = 0.5*(1+erf((x-mu)/sqrt(2var)))
            prob = cdf(x+deltax)-cdf(x-deltax)
        """
        #TODO: fix deltax to be retrieved from kwargs    
        deltax = self.deltax

        xplus  = (x+deltax-mu)/(var*2)**0.5
        xless  = (x-deltax-mu)/(var*2)**0.5
        px     = 0.5*(scipy.special.erf(xplus)-scipy.special.erf(xless))
        return np.clip(px,1e-15,np.inf)
    
    def jointpdf(self,model,Xn,Yn,Varn,**kwargs):
        """
        model:
            mapping class that generates the predictions for Xn
        Xn:
            tp test points as a numpy array [tp, 2]
        Yn: 
            mp measurement points of all n wireless measurements as a numpy array [mp, n]
        Varn: 
            mp var measurement points of all n wireless measurements as a numpy array [mp, n]
        kwargs:
            log: if True, ouputs the log joint probability instead - default False
            
        """
        logFlag         = kwargs.get('log',False)
        deltax          = self.deltax
        confidence_zero = self.confidence_zero
        pzero           = self.pzero
        factor_alpha    = kwargs.get('factor_alpha',self.falpha)
        alpha           = kwargs.get('alpha',None)
        get_alpha       = kwargs.get('get_alpha',False)
        cor             = kwargs.get('cor',None)
        ignore_cor      = kwargs.get('ignore_cor',True)
        get_cor         = kwargs.get('get_cor',False)
        get_px          = kwargs.get('get_px',False)        
        verbose         = kwargs.get('verbose',False)        

        if Yn.ndim==1:
            measurement = Yn.copy().reshape(1,len(Yn))
            #measure_var = Varn.copy().reshape(1,len(Varn))
        else:
            measurement = Yn.copy()
            #measure_var = Varn.copy()
        
        ypredict, varpredict = model.predict(Xn)       
        assert measurement.shape[1]==ypredict.shape[1]
        
        n_testp  = measurement.shape[0]
        n_points = ypredict.shape[0]
        n_ap     = measurement.shape[1]
    
        if get_px:
            #create a list of arrays to contain a px for each ap
            pX = [np.zeros((n_points,n_testp)) for i in range(n_ap)] 

        if get_alpha:
            Alpha = list()
        
        if get_cor:
            Cor = list()

        log_pXV  = np.zeros((n_points,n_testp))
    
        if varpredict.shape[1] == 1:
            var  = varpredict.copy()
            
        for i in range(n_testp):
            xs   = np.reshape(measurement[i,:],(1,n_ap))
            mus  = ypredict
            if varpredict.shape[1] > 1:
                var  = varpredict[:,i:i+1]
        
            px      = (1-pzero)*self.prob(xs,mus,var,deltax=deltax)
            #except:
            #    eprint('ERROR')
            #    px = np.zeros_like(xs)

            px_zero = np.zeros_like(xs)         # Probability if the measurement is zero
            px_zero[np.where(xs==0)] = pzero  
            px      = px + px_zero              #Total probability 

            if get_px:
                for ii in range(n_ap):                
                    pX[ii][:,i] = px[:,ii]
            
            #Zero measurements are not trusted the same necessarily 
            # - if confidence_zero = 1 they are
            #cz = confidence_zero

            if confidence_zero == 0:            
                cz = (1.*(n_ap-np.sum(measurement[i,:]==0)))/(np.sum(measurement[i,:]==0)*1.)
            else:
                if confidence_zero is None:            
                    cz = 1./(np.sum(measurement[i,:]==0)*1.)
                else:
                    cz = confidence_zero
            
            if alpha is None:
                alpha = np.ones(n_ap) #(np.max(px,axis=0)-np.mean(px,axis=0))/np.mean(px,axis=0) 

            if get_alpha:
                Alpha.append(alpha)

            if (cor is None) and (not ignore_cor):
                COR = np.ones((n_ap,n_ap))

                for ii in range(n_ap):
                    for jj in range(ii+1,n_ap):
                        COR[ii,jj] = np.sum(px[:,ii]*px[:,jj])/(0.5*np.sum(px[:,ii]**2+px[:,jj]**2))
                        COR[jj,ii] = COR[ii,jj]
    
                try:
                    COR_inv = np.power(COR,-1)
                except:
                    eprint("[WARN] No stable inverse, pinverse used")
                    COR_inv = np.linalg.pinv(COR)
            
                cor = np.dot(COR_inv,np.ones((COR.shape[0],1))).flatten()
            
            if get_cor:
                Cor.append(cor)

            alpha[np.where(measurement[i,:]==0)] *= cz   #all zeros get one vote

            if not ignore_cor:
                alpha = alpha*cor   #APs correlations
            alpha = factor_alpha*alpha/np.sum(alpha)     #normalize
            log_px = np.sum(alpha*np.log(px),axis=1,keepdims=True)
            log_pXV[:,i] = log_px.flatten()


        return_vector = list()        
            
        if logFlag:
            return_vector.append(np.clip(log_pXV,-200,200))

        else:
            return_vector.append(np.exp(log_pXV))

        if get_px:
            return_vector.append(pX)

        if get_alpha:
            return_vector.append(Alpha) 

        if get_cor:
            return_vector.append(Cor)

        #if only one element is requested (logpx or px) then the output is not a list
        if len(return_vector) == 1:
            return return_vector[0]
        else:
            return return_vector
        



















