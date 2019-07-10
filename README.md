# WifiLocalization

Wireless signal strength based localization

## Installation

clone this git repository and add it to your path:

    git clone https://github.com/RMiyagusuku/wifi-localization.git ~/WifiLocalization
    echo 'export PYTHONPATH=$PYTHONPATH:~/WifiLocalization' >> ~/.bashrc

install levmarmodule optimization for python

    cd ~/WifiLocalization/Localization/sensormodel/pathloss/levmarmodule
    sudo python setup.py build     # to compile
    sudo python setup.py install   # to install 

Wrappers for ROS  available at
    
    https://github.com/RMiyagusuku/ros-wifi-localization


## Tested Platform

    Ubuntu 14.04.5
    Python 2.7

## Libraries dependencies

    GPy at https://github.com/SheffieldML/GPy
    Numpy 1.11.1
    Scipy 0.18.0



## Related publications:
> **Renato Miyagusuku**, Atsushi Yamashita and Hajime Asama: "Data Information Fusion from Multiple Access Points for WiFi-based Self-localization", IEEE Robotics and Automation Letters, Vol. 4, No. 2, pp. 269-276, April 2019. [[doi:10.1109/LRA.2018.2885583]](https://doi.org/10.1109/LRA.2018.2885583) <br/>
> **Renato Miyagusuku**, Atsushi Yamashita and Hajime Asama:"Precise and accurate wireless signal strength mappings using Gaussian processes and path loss models", Robotics and Autonomous Systems, February 2018. [[doi:10.1016/j.robot.2018.02.011]](https://doi.org/10.1016/j.robot.2018.02.011) <br/>
> **Renato Miyagusuku**, Atsushi Yamashita and Hajime Asama: "Gaussian Processes Mappings Improvements Using Path Loss Models for Wireless Signals-based Localization", in: IEEE/RSJ International Conference on Intelligent Robots and Systems, 2016, pp. 4610-4615. [[doi:10.1109/IROS.2016.7759678]](https://doi.org/10.1109/IROS.2016.7759678) <br/> * This work employs a previous python implementation available [here](../IROS2016/)

### Maybe also of interest
WLRF: WiFi and range data fusion <br/>
Main implementations for fusing WiFi and range data: 
[Modified amcl](https://github.com/RMiyagusuku/navigation/tree/indigo-devel/amcl), [wlrf ROS package](https://github.com/RMiyagusuku/wifi-localization) <br/>
How to generate additional rosbags for testing global localization and the kidnapped robot problem can be found [here](../datasets/)
> **Renato Miyagusuku**, Yiploon Seow, Atsushi Yamashita and Hajime Asama: "Fast and Robust Localization using Laser Rangefinder and WiFi Data", in: IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems, 2017, pp. 111-117. [[doi:10.1109/MFI.2017.8170415]](https://doi.org/10.1109/MFI.2017.8170415)


## Questions

Please mail directly to miyagusuku at robot.t.u-tokyo.ac.jp


