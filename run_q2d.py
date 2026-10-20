import numpy as np
from A2codes import synExperimentsKernel

train_acc, test_acc = synExperimentsKernel()
print("TRAIN (rows: Linear, Poly2, Poly3, Gauss1, Gauss0.5 | cols: SqHinge M1 M2 M3, Hinge M1 M2 M3)")
print(np.round(train_acc, 3))
print("TEST")
print(np.round(test_acc, 3))