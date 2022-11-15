import numpy as np
a = np.array([1, 1, 2, 3, 5, 8])  # Start with an existing NumPy array
from multiprocessing import shared_memory

shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
b[:] = a[:] 
print(shm.name)

