import numpy as np
from multiprocessing import shared_memory

existing_shm = shared_memory.SharedMemory(name='psm_1e75b936')
c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)

print(c)

existing_shm.close()