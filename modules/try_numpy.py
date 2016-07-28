
import numpy as np 
a = np.arange(16)
b = a.reshape((2,2,4))
c = b.reshape(4,-1)
print a,b
print c









