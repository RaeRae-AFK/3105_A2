from A2codes import synExperimentsRegularize
import numpy as np
train_acc, test_acc = synExperimentsRegularize()
print(np.round(train_acc, 3)); print(np.round(test_acc, 3))

