## Refs
https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

https://kr.mathworks.com/help/matlab/matlab_external/call-matlab-functions-asynchronously-from-python.html

https://kr.mathworks.com/help/matlab/import_export/share-memory-between-applications.html

https://stackoverflow.com/questions/65622900/shared-workspace-between-matlab-and-python

https://kr.mathworks.com/help/matlab/matlab_external/table-of-mex-file-source-code-files.html

## install matlab engine for python

(test1) neurlab@NeuRLab-DL4:/usr/local/MATLAB/R2018b/extern/engines/python$ sudo python setup.py install --prefix /home/neurlab-dl4/matlab_engine/matlab2018b

## test
python
>>> import sys
>>> sys.path.append("/home/neurlab-dl4/matlab_engine/matlab2018b/lib/python2.7/site-packages/")
>>> import matlab # worked!
