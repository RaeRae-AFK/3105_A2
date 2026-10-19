import numpy as np
from A2codes import dualHinge, dualClassify, minHinge
from A2helpers import generateData, linearKernel, polyKernel

np.random.seed(1)
X, y = generateData(n=100, gen_model=1)
Xtest, ytest = generateData(n=500, gen_model=1)
lamb = 0.01

# Runs and gives sensible accuracy
k = lambda X1, X2: polyKernel(X1, X2, 2)
a, b = dualHinge(X, y, lamb, k)
yhat = dualClassify(Xtest, a, b, X, y, lamb, k)
print("shape:", yhat.shape)                       # should be (500, 1)
print("test acc:", np.mean(yhat.ravel() == ytest.ravel()))   # should be high, ~0.95+

# Debug hint from the assignment: linear kernel should match Q1(b)
a, b = dualHinge(X, y, lamb, linearKernel)
w_dual = X.T @ (y.reshape(-1, 1) * a) / lamb
w_primal, w0 = minHinge(X, y, lamb)
print("w dual:  ", w_dual.ravel(), " b:", b)
print("w primal:", w_primal.ravel(), " w0:", w0)