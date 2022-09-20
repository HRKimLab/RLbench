import sys
# this seems not work
sys.path.append("/home/neurlab-dl4/matlab_enging/matlab2018b")
# this works
sys.path.append("/home/neurlab-dl4/matlab_engine/matlab2018b/lib/python2.7/site-packages/")

import matlab
print(matlab.__file__)

import matlab.engine
eng = matlab.engine.start_matlab()
print("MATLAB engine started")

tf = eng.isprime(37)
print(tf)
