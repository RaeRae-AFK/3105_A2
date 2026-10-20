import numpy as np
from A2codes import adjSqHinge, adjHinge, adjClassify, minSqHinge, minHinge
from A2helpers import generateData, linearKernel, polyKernel, plotAdjModel

np.random.seed(1)
X, y = generateData(n=100, gen_model=1)
Xtest, ytest = generateData(n=500, gen_model=1)
lamb = 0.01

# Debug hint: with a linear kernel, w = X^T a should match Q1
a, a0 = adjSqHinge(X, y, lamb, linearKernel)
w, w0 = minSqHinge(X, y, lamb)
print("SqHinge adjoint w:", (X.T @ a).ravel(), " a0:", a0)
print("SqHinge primal  w:", w.ravel(), " w0:", w0)

a, a0 = adjHinge(X, y, lamb, linearKernel)
w, w0 = minHinge(X, y, lamb)
print("Hinge adjoint w:", (X.T @ a).ravel(), " a0:", a0)
print("Hinge primal  w:", w.ravel(), " w0:", w0)

# Shape and accuracy check with a nonlinear kernel
k = lambda X1, X2: polyKernel(X1, X2, 2)
a, a0 = adjHinge(X, y, lamb, k)
yhat = adjClassify(Xtest, a, a0, X, k)
print("shape:", yhat.shape)   # should be (500, 1)
print("test acc:", np.mean(yhat.ravel() == ytest.ravel()))

# Visual check on the circle data
X2, y2 = generateData(n=100, gen_model=2)
a, a0 = adjHinge(X2, y2, lamb, k)
plotAdjModel(X2, y2, a, a0, k, adjClassify)