# WifiLocalization

Wireless signal strength based localization

## Installation

clone this git repository and add it to your path:

    git clone git@github.com:RMiyagusuku/wifi-localization.git ~/WifiLocalization
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

## Questions

Please mail directly to miyagusuku at robot.t.u-tokyo.ac.jp


