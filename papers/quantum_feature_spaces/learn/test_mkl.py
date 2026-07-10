from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import numpy as np

X = [[1,2],
     [2,3],
     [3,4],
     [1,1]]

y = [[1,2],
     [2,3]]

X = np.array(X)

ker = rbf_kernel(y, X, 2)

print(ker)