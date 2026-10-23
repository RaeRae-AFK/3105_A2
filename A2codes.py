#Rachel Chung
#1010333019
#Sheethal Thachet
#101332602

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from A2helpers import generateData, linearKernel, polyKernel, gaussKernel
import cvxopt

#Question 1a
def minSqHinge(X, y, lamb):
    n, d = X.shape #X is an nxd matrix. n is data points, d features each
    y = y.reshape(-1) #reshape/flatten so it stays below n

    #computes loss and gradient for a given guess of w and w0
    def obj(theta):
        w, w0 = theta[:d], theta[d] #split guess into w and w0
        r = np.maximum(0, 1 - y * (X @ w + w0)) #how much each point violates the marigin (if fine 0)
        f = np.sum(r ** 2) + 0.5 * lamb * w @ w #total loss: squared violations + penalty on w
        g_w = -2 * X.T @ (r * y) + lamb * w #slope of loss with respect to w
        g_w0 = -2 * np.sum(r * y) #~~~~~~ w0
        return f, np.append(g_w, g_w0)

    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B") #start at all zeros and let scipy find w, w0 with lowest loss
    return res.x[:d].reshape(-1, 1), res.x[d] #return w as column and w0

# Question 1b
def minHinge(X, y, lamb, stabilizer=1e-5):
    n, d = X.shape
    y = y.reshape(-1, 1).astype(float) #labels as column of decimals

    P = np.zeros((d + 1 + n, d + 1 + n)) #only w is squared in the loss
    P[:d, :d] = lamb * np.eye(d)
    P += stabilizer * np.eye(d + 1 + n) #boost to keep solver stable
    q = np.vstack([np.zeros((d + 1, 1)), np.ones((n, 1))]) #q adds up all the xi values

    G1 = np.hstack([np.zeros((n, d + 1)), -np.eye(n)]) #every xi must be >= 0
    G2 = np.hstack([-y * X, -y, -np.eye(n)]) # xi must cover each point's marigin violation
    G = np.vstack([G1, G2]) #all rules combine together
    h = np.vstack([np.zeros((n, 1)), -np.ones((n, 1))]) #right hand side of the rules

    cvxopt.solvers.options["show_progress"] = False # don't print solver output
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h))
    u = np.array(sol["x"])
    return u[:d], u[d, 0]

# Question 1c
def classify(Xtest, w, w0):
    return np.where(Xtest @ w + w0 >= 0, 1, -1).reshape(-1, 1) #predict +1 if x^t w + w0 >= 0, otherwise -1 as a column

#Quetsion 1d
def synExperimentsRegularize():
    n_runs = 100 #settings rules based on the pdf
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]


    train_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs]) #empty arrays to store accuracy
    test_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    np.random.seed(51) #seed with our group number

    def acc(X, y, w, w0): #fraction of points predicted correctly
        return np.mean(classify(X, w, w0).ravel() == y.ravel())

    for r in range(n_runs): #try every combo of run, lamda and data model
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list): #make new training and test data
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                w, w0 = minSqHinge(Xtrain, ytrain, lamb) #train with squared hinge, save accuracies
                train_acc_sq_hinge[i, j, r] = acc(Xtrain, ytrain, w, w0)
                test_acc_sq_hinge[i, j, r] = acc(Xtest, ytest, w, w0)

                w, w0 = minHinge(Xtrain, ytrain, lamb) #train with hinge, save accuracies
                train_acc_hinge[i, j, r] = acc(Xtrain, ytrain, w, w0)
                test_acc_hinge[i, j, r] = acc(Xtest, ytest, w, w0)

    #avg of over 100 runs, then put SqHinge and hinge side by side in 4x6 table
    train_acc = np.hstack([train_acc_sq_hinge.mean(axis=2), train_acc_hinge.mean(axis=2)])
    test_acc = np.hstack([test_acc_sq_hinge.mean(axis=2), test_acc_hinge.mean(axis=2)])
    return train_acc, test_acc

#Question 1e in pdf

#----------------------------------
# Question 2
#Test
#print("h")

#2a: adjoint form of regularized squared hinge lkoss
def adjSqHinge(X, y, lamb, kernel_func):
    n = X.shape[0]
    #Flattening the labels to shape (n,) 
    y = y.reshape(-1).astype(float)
    #nxn kernel matrix of the training data
    K = kernel_func(X, X)

    #Returning the loss value and its gradient so that the optimizer doesn't have to estimate gradients numerically
    def obj(theta):
        #takes all parameters into one vector
        a, a0 = theta[:n], theta[n]

        #Score of each traning point (before a0)
        Ka = K @ a
        #Hinge residual max per point
        r = np.maximum(0, 1 - y * (Ka + a0))

        #Squaring hinge loss + regularizer (lamb/2) a^T K a
        f = np.sum(r ** 2) + 0.5 * lamb * a @ Ka

        #Greadients
        g_a = -2 * K @ (r * y) + lamb * Ka
        g_a0 = -2 * np.sum(r * y)

        return f, np.append(g_a, g_a0)

    #Starting from all 0's --> jac=True will tell the scipy tha obj also returns the gradient
    res = minimize(obj, np.zeros (n + 1), jac=True, method="L-BFGS-B")

    #Unpacking the solution (a is an nx1 column vector, a0 is a scalar)
    return res.x[:n].reshape(-1, 1), res.x[n]

#2b: Regularized hinge loss, solved as a quadratic program
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    #Label is a nx1 col vector 
    y = y.reshape(-1,1).astype(float)
    #nxn kernel matric
    K = kernel_func(X, X)

    P = np.zeros((2 * n + 1, 2 * n + 1))
    P[:n, :n] = lamb * K
    P += stabilizer * np.eye(2 * n + 1)

    #0 for a and a0, 1 fore each xi --> giving sum of xi
    q = np.vstack([np.zeros((n + 1, 1)), np.ones((n, 1))])

    #Constraint 1
    G1 = np.hstack([np.zeros((n, n + 1)), -np.eye(n)])

    #Constrint 2
    G2 = np.hstack([-y * K, -y, -np.eye(n)])

    #Stacking both contraints into one (G)
    G = np.vstack([G1, G2])
    h = np.vstack([np.zeros((n, 1)), -np.ones((n, 1))])

    #Solving the QP
    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h))
    u = np.array(sol["x"])

    #Returning a, a0 and xi is discarded
    return u[:n], u[n, 0]

#2c: Predict the labels for the test points using the adjoing model
def adjClassify(Xtest, a, a0, X, kernel_func):
    #one score per test point --> copmutes for all point at onec
    z = kernel_func(Xtest, X) @ a.reshape(-1, 1) + a0

    #convertng scores to labels (returns a mx1 vector)
    return np.where(z >= 0, 1, -1).reshape(-1, 1)

#2d: Compare 5 kernels x 3 datasets x 2 losses, averaged over 10 runs
def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001

    #Each kernel is wrapped as a function of (X1, X2) (so they can be called the same wasy)
    kernel_list = [linearKernel, lambda X1, X2: polyKernel(X1, X2, 2), lambda X1, X2: polyKernel(X1, X2, 3), lambda X1, X2: gaussKernel(X1, X2, 1.0), lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]

    #Accuracy storage --> indexed as [kernel, dataset, run]
    train_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])

    #Fixing the random seed so results are reproducible (we are group A2-51)
    np.random.seed(51)

    #Helper: fraction of points whose predictied label matches the true label
    def acc(Xeval, yeval, a, a0, Xtrain, kernel):
        return np.mean(adjClassify(Xeval, a, a0, Xtrain, kernel).ravel() == yeval.ravel())

    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                #Squared hinge model: train, then measuring train an test accuracy
                a, a0 = adjSqHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_sq_hinge[i, j, r] = acc(Xtrain, ytrain, a, a0, Xtrain, kernel)
                test_acc_sq_hinge[i, j, r] = acc(Xtest, ytest, a, a0, Xtrain, kernel)

                #Hinge model: train, then measuring train and test accuracy
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = acc(Xtrain, ytrain, a, a0, Xtrain, kernel)
                test_acc_hinge[i, j, r] = acc(Xtest, ytest, a, a0, Xtrain, kernel)

    #Average over the runs, then placing SqHinge beside Hinge to give 5x6
    train_acc = np.hstack([train_acc_sq_hinge.mean(axis=2), train_acc_hinge.mean(axis=2)])
    test_acc = np.hstack([test_acc_sq_hinge.mean(axis=2), test_acc_hinge.mean(axis=2)])
    return train_acc, test_acc

#-------------------------------------
# Question 3
#3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0] #number of points
    y = y.reshape(-1, 1).astype(float) #labels as a column of decimals
    K = kernel_func(X, X) #similarity between every pair of points

    P = (y @ y.T) * K / lamb + stabilizer * np.eye(n) #P the squared part of the objective
    q = -np.ones((n, 1)) #q: the linear part
    G = np.vstack([-np.eye(n), np.eye(n)]) #every alpha must be between 0 and 1
    h = np.vstack([np.zeros((n, 1)), np.ones((n, 1))]) #right hand side 0 and 1
    A = y.T #must add up to 0
    b_eq = np.zeros((1, 1))

    cvxopt.solvers.options["show_progress"] = False #dont print solver output
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h),
                            cvxopt.matrix(A), cvxopt.matrix(b_eq))
    a = np.array(sol["x"]) # the solution thingy

    i = np.argmin(np.abs(a - 0.5)) #find hte offset b using the point whose alpha is closest to 0.5
    b = y[i, 0] - (K[i] @ (y * a)).item() / lamb 
    return a, b

#3b
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    y = y.reshape(-1, 1) # labels as columb
    z = kernel_func(Xtest, X) @ (y * a) / lamb + b #score each test point by its similarity to the training points
    return np.where(z >= 0, 1, -1).reshape(-1, 1) #predict +1 if score is >= 0 otherwise -1

#3c
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy() #load the digit images
    X = train_data[:, 1:] / 255. #pixel values, scaled to between 0 and 1
    y = train_data[:, 0][:, None] #first column is the digit as a column
    y[y == 4] = -1 #digit 4 -> -1
    y[y == 9] = 1 #9-> +1

    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)]) #empty array to store accuracy for every fold, lambda and kernel

    np.random.seed(51) #seed group number thingy
    folds = np.array_split(np.random.permutation(X.shape[0]), k) #shuffle point nums and split them into k groups

    for i, lamb in enumerate(lamb_list): #every combination of lambda and kernel
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                val_idx = folds[l]
                train_idx = np.concatenate([folds[m] for m in range(k) if m != l])
                Xtrain, ytrain = X[train_idx], y[train_idx]
                Xval, yval = X[val_idx], y[val_idx]
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func) #train on training folds, predict validation fold
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                cv_acc[l, i, j] = np.mean(yhat.ravel() == yval.ravel()) #fraction correct

    cv_acc = cv_acc.mean(axis=0) #avg accuracy over k folds
    i_best, j_best = np.unravel_index(np.argmax(cv_acc), cv_acc.shape) #lamda and kernel w/ highest avg accuracy
    return cv_acc, lamb_list[i_best], kernel_list[j_best]








