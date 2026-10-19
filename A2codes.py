import numpy as np
import pandas as pd
from scipy.optimize import minimize
from A2helpers import generateData
import cvxopt

#Question 1a
def minSqHinge(X, y, lamb):
    n, d = X.shape
    y = y.reshape(-1)

    def obj(theta):
        w, w0 = theta[:d], theta[d]
        r = np.maximum(0, 1 - y * (X @ w + w0))
        f = np.sum(r ** 2) + 0.5 * lamb * w @ w
        g_w = -2 * X.T @ (r * y) + lamb * w
        g_w0 = -2 * np.sum(r * y)
        return f, np.append(g_w, g_w0)

    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:d].reshape(-1, 1), res.x[d]

# Question 1b
def minHinge(X, y, lamb, stabilizer=1e-5):
    n, d = X.shape
    y = y.reshape(-1, 1).astype(float)

    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = lamb * np.eye(d)
    P += stabilizer * np.eye(d + 1 + n)
    q = np.vstack([np.zeros((d + 1, 1)), np.ones((n, 1))])

    G1 = np.hstack([np.zeros((n, d + 1)), -np.eye(n)])
    G2 = np.hstack([-y * X, -y, -np.eye(n)])
    G = np.vstack([G1, G2])
    h = np.vstack([np.zeros((n, 1)), -np.ones((n, 1))])

    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h))
    u = np.array(sol["x"])
    return u[:d], u[d, 0]

# Question 1c
def classify(Xtest, w, w0):
    return np.where(Xtest @ w + w0 >= 0, 1, -1).reshape(-1, 1)

#Quetsion 1d
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    np.random.seed(0)

    def acc(X, y, w, w0):
        return np.mean(classify(X, w, w0).ravel() == y.ravel())

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                w, w0 = minSqHinge(Xtrain, ytrain, lamb)
                train_acc_sq_hinge[i, j, r] = acc(Xtrain, ytrain, w, w0)
                test_acc_sq_hinge[i, j, r] = acc(Xtest, ytest, w, w0)

                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = acc(Xtrain, ytrain, w, w0)
                test_acc_hinge[i, j, r] = acc(Xtest, ytest, w, w0)

    train_acc = np.hstack([train_acc_sq_hinge.mean(axis=2), train_acc_hinge.mean(axis=2)])
    test_acc = np.hstack([test_acc_sq_hinge.mean(axis=2), test_acc_hinge.mean(axis=2)])
    return train_acc, test_acc

#Question 1e in pdf

#----------------------------------
# Question 2


#-------------------------------------
# Question 3
#3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    y = y.reshape(-1, 1).astype(float)
    K = kernel_func(X, X)

    P = (y @ y.T) * K / lamb + stabilizer * np.eye(n)
    q = -np.ones((n, 1))
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.vstack([np.zeros((n, 1)), np.ones((n, 1))])
    A = y.T
    b_eq = np.zeros((1, 1))

    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h),
                            cvxopt.matrix(A), cvxopt.matrix(b_eq))
    a = np.array(sol["x"])

    i = np.argmin(np.abs(a - 0.5))
    b = y[i, 0] - (K[i] @ (y * a)).item() / lamb 
    return a, b

#3b
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    y = y.reshape(-1, 1)
    z = kernel_func(Xtest, X) @ (y * a) / lamb + b
    return np.where(z >= 0, 1, -1).reshape(-1, 1)

#3c
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1

    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])

    np.random.seed(0)
    folds = np.array_split(np.random.permutation(X.shape[0]), k)

    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                val_idx = folds[l]
                train_idx = np.concatenate([folds[m] for m in range(k) if m != l])
                Xtrain, ytrain = X[train_idx], y[train_idx]
                Xval, yval = X[val_idx], y[val_idx]
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                cv_acc[l, i, j] = np.mean(yhat.ravel() == yval.ravel())

    cv_acc = cv_acc.mean(axis=0)
    i_best, j_best = np.unravel_index(np.argmax(cv_acc), cv_acc.shape)
    return cv_acc, lamb_list[i_best], kernel_list[j_best]








