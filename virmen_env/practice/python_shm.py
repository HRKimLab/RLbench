import numpy
import sys
import shared_memory

# setup
sys.path.append(r'C:/download/python')
from shared_memory import *
shared_memory.set_shared_memory_path(r'C:/download/data')

# receive
data = shared_memory.get_shared_memory_data()

# manipulate
new_data = numpy.sin(data)

# share
shared_memory.set_shared_memory_data(new_data)