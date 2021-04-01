import numpy as np

"""
Verifying the equation for generating cov matrix
sum over all sigma points (wi * outer(Xi, Xi))
is identical to matrix multiplication of X outer X 
"""
w = np.array([1,2,3,4,1])
X = np.array([[1,2,3,4,5] for i in range(3)])
Arr = np.zeros((3,3))
for i in range(5):
    xi = X[:,i]
    Arr += w[i] * np.outer(xi, xi) # each col outer product
print(Arr)

Arr2 = np.zeros((3,3))
print((w*X)@X.T) # (3*5)*5*3 with first one passing a broadcast multiplication