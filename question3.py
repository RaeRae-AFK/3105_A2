import numpy as np
import pandas as pd
from A2codes import cvMnist
from A2helpers import linearKernel, polyKernel, gaussKernel

lamb_list = [0.01, 0.1, 1.0]
kernel_list = [linearKernel,
               lambda X1, X2: polyKernel(X1, X2, 2),
               lambda X1, X2: gaussKernel(X1, X2, 5.0)]
cv_acc, best_lamb, best_kernel = cvMnist("/Users/rachelchung/Downloads/A2files", lamb_list, kernel_list)
print(np.round(cv_acc, 3), best_lamb)