## Refs
https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

https://kr.mathworks.com/help/matlab/matlab_external/call-matlab-functions-asynchronously-from-python.html

https://kr.mathworks.com/help/matlab/import_export/share-memory-between-applications.html

https://stackoverflow.com/questions/65622900/shared-workspace-between-matlab-and-python

https://kr.mathworks.com/help/matlab/matlab_external/table-of-mex-file-source-code-files.html

## install matlab engine for python
```
(test1) neurlab@NeuRLab-DL4:/usr/local/MATLAB/R2018b/extern/engines/python$ sudo python setup.py install --prefix /home/neurlab-dl4/matlab_engine/matlab2018b
```
## test

```
python
>>> import sys
>>> sys.path.append("/home/neurlab-dl4/matlab_engine/matlab2018b/lib/python2.7/site-packages/")
>>> import matlab # worked!
```
## python-python interaction

- 서로 다른 두 터미널에서 shared_memory.SharedMemory 이용하면 python 간 interaction 가능하나, 파이썬 파일로 하면 directory error로 인해 send만 되고 call은 안됨
- /dev/shm에서 존재하는 shared memory name 확인 가능
- a[:] = a[:]+1 과 같이 직접 assign 해줄 때는 array의 id가 바뀌지 않으나, a = a+1과 같은 연산을 할 때에는 id가 바뀜. (같은 뜻인데 a+=1은 id 안 바뀜)

## matlab-matlab interaction

- Use memmapfile function; it makes mapped memory (shared memory) for a file
- It can exchange array data types (ex.[4 1])
```
>>> m = memmapfile(filename, 'Writable', true, 'Format', {'uint32' [4 1] name});
>>> m.data(1).name
```
