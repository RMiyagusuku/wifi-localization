class SensorModel(object):   
    """
    Sensor model
    """
    def __init__(self):
        super(SensorModel,self).__init__()
        self.name = 'Sensor Model'

    ####################          To implement by derived classes          ####################
    def sample(self,measurements):
        """
        (Optional) For generative models, given sensor measurements, sample the model 
        """
        raise NotImplementedError


    def sensor_update(self,particles,measurements,**kwargs):
        """
        Computes the weights for the particles given the measurements 
        """
        raise NotImplementedError   

