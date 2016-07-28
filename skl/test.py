import numpy as np
from scipy.spatial.distance import cdist
# A1 = [[0.5,0.2,0.3], [0.2,0.3,0.5]]
# A = np.array(A1)
# print A[1,:].max()

# for i in range(2,10):
# 	print i

# print max(i*2 for i in range(10))
# print np.array([i*2 for i in range(10)]).max()
# print np.array([i*2 for i in range(10)]).argmax()
# a = [0.5,0.2,0.3]
# print sum([i*2 for i in range(10) if i>8])

a = np.zeros((3,4))
b = np.ones((3,4))
c = [[0] * 5] * 5
print sum(sum(b))
print sum([3,2])
# c = [[0.5,0.2,0.3]]
# e = [[0.2,0.3,0.5]]

# f = sum(cdist(c,e, 'chebyshev'))
# d = cdist(a,b, 'chebyshev')
# print d,f