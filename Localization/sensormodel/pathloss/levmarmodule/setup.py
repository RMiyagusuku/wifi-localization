from distutils.core import setup, Extension

module1 = Extension('levmar',
                    sources = ['levmarmodule.c'],
                    libraries = ['m','levmar','lapack','blas','f2c'],
                    extra_compile_args = ['-Ofast'],
                    )

setup (name = 'levmar',
       version = '1.1',
       description = 'Levmar package',
       ext_modules = [module1],)

#sudo python setup.py build          # to compile
#sudo python setup.py install   # to install 
#to completely uninstall remove manually levmar-1.0.egg-info from usr/local/lib/python2.7/dist-packages or pip uninstall levmar
