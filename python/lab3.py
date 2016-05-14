
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
#
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
#
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
#
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
#
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
from sklearn import decomposition
from matplotlib.colors import ColorConverter

# ## Bayes classifier functions to implement
#
# The lab descriptions state what each function should do.

# ###Miscelleneous Functions
#
# Tests if a matrix A is definite positive
# in:      A - matrix
def is_positive_definite(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M+M.transpose()) > 0)


# If matrix has NAN elements, substitute them with zeros
# in:      A - matrix
def set_Nan_to_zero(A):
    M = np.matrix(A)
    A[np.isnan(A)] = 0

# checks If matrix is symmetric
# in:      A - matrix
def is_symmetric(A):
    M = np.matrix(A)
    return np.allclose(A.transpose(1, 0), A)

# convertes a matrix to symmetric, positive and definite matrix
# in:      A - matrix
def convert_to_symmetric_positive_definite(A):
    vals, vecs = np.linalg.eigh(A)
    return A + vecs - 6 * np.identity

# in:    cls - class number
#          x - 1 x d vector of 1 data point to be classified
#         mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
# out:     L - d x d upper triangular matrix or inverse of sigma
# out:     x - N x 1 transposed
def computeY(cls, x, mu, sigma, covdiag = True):
    L = np.zeros(sigma.shape)
    x = 0
    if not covdiag:
        A = sigma[:,:]
        b = np.transpose(np.subtract(x, mu))
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L,b)
        # x = np.linalg.solve(L.getH(),y)
        x = np.linalg.solve(np.transpose(L), y)
#     print("L ")
#     print(L.shape)
#     print(type(L))
#     print("x ")
#     print(x.shape)
#     print(type(x))
    return L, np.transpose(x)

# in:              x - 1 x d vector of 1 data point to be classified
#              cls   - class number
#              prior - C x 1 vector of class priors
#                 mu - C x d matrix of class means
#              sigma - d x d x C matrix of class covariances
#                  L - d x d x C matrix of class covariances
#                  y - d x d x C matrix of class covariances
# out:  logPosterior - N x 1 class predictions for test points
def computeLogPosterior(x, cls, prior,mu,sigma, L, y, covdiag=True):
    # lnl = 2 * np.sum(np.log(np.diag(L)))
    lnl = np.log(np.linalg.det(sigma))
    if covdiag:
        # print("Sub ", np.subtract(x, mu).shape)
        # print("L ", L.shape)
        # print("dot1 ", np.dot(np.subtract(x, mu), L).shape)
        # print("tanspose dot1 ", np.transpose(np.dot(np.subtract(x, mu), L)).shape)
        # print("Y ", y.shape)

        # print("lnl ", lnl)
        logPosterior = \
            -1 / 2 * lnl \
            - 1 / 2 * np.dot(np.dot(np.subtract(x, mu),L), np.transpose(y)) \
            + np.log(prior)
    else:
        logPosterior = \
            -1/2 * lnl \
            - 1/2 * np.dot(np.subtract(x, mu), y) \
            + np.log(prior)

#         - 1/2 * np.dot(np.subtract(x, mu), y) \

#     print(type(y))
#     print(type(np.subtract(x, mu)))
#     print("type of multiplication")
#     print((np.subtract(x, mu) * y).shape)
#     print(type(np.subtract(x, mu) * y))
    return logPosterior

# Note that you do not need to handle the W argument for this part
# in: labels - N x 1 vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels,W=None):
    # Your code here
    #### sort X by class
    classes = np.unique(labels)
    classesNo = classes.shape[0]
    dataNo = labels.shape[0]
    if W is None:
        l = [[labels[i] for i in range(dataNo) if labels[i] == clas] \
             for clas in classes]
        prior = np.zeros(len(l))
        prior = [len(l[i]) / dataNo for i in range(len(l))]
    else:
        prior = np.empty(classesNo)
        for i in range(classesNo):
            prior[labels[i]] += W[i]

    # print(prior)
    # labels = [int(label) for label in labels]
    # print(np.bincount(labels)/len(labels))

    return prior

# Note that you do not need to handle the W argument for this part
# in:      X - N x d matrix of N data points
#     labels - N x 1 vector of class labels
#     W      - N x 1 vector of weights
# out:    mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
def mlParams(X,labels,W=None):
    # Your code here
    labels = [int(label) for label in labels]
    classes = np.unique(labels)

    classesNo = classes.shape[0]
    dims = X.shape[1]
    dataNo = X.shape[0]
    if W is None:
        W = np.ones(dataNo)

    mu = np.zeros((classesNo, dims));
    sigma = np.zeros((dims, dims, classesNo))

    # print("mlParams------------")
    # print(X.shape)
    # print(mu.shape)
    # print(sigma.shape)
    # print(classes)
    # print("mlParams-2----------")

    #### sort X by class
    Xt = np.transpose(X)
    # print("Xt ", Xt.shape)
    # print("Xt ", Xt)
    # Wt = np.transpose(W)
    # print(Wt)
    Xw = np.transpose(np.multiply( W, Xt))
    # print("Xw ", Xw.shape)
    # print("Xw ", Xw)

    Xk = np.asarray([[Xw[i] for i in range(dataNo) if labels[i] == clas] \
                     for clas in classes])

    Xk2 = np.asarray([[X[i] for i in range(dataNo) if labels[i] == clas] \
                     for clas in classes])

    Wk = np.asarray([[W[i] for i in range(dataNo) if labels[i] == clas] \
                     for clas in classes])
    # print(Xk2.shape)
    # print(Wk.shape)
    # print(mu.shape)
    # print(Xk[0])
    # print(type(Xk))
    # print(labels[0])
    # print("mlParams------------")
    for cls in range(classesNo):
        for d in range(dims):
            for n in range(Xk2.shape[1]):
                # print(cls, d, n)
                mu[cls, d] += Xk2[cls, n, d] * Wk[cls, n]  # (mu[cls, d] / Wks[cls]

    Wks = np.sum(Wk, axis=1)
    for cls in range(classesNo):
        for d in range(mu.shape[1]):
            mu[cls, d] = mu[cls, d] / Wks[cls]

    # mu = np.transpose(np.sum(Xk, axis=1)) #/ np.sum(W)
    # mu = np.sum(Xk, axis=1)  # / np.sum(W)
    # print(mu.shape)




    # print(Wks)
    # mum = np.transpose(np.divide(mu,Wks))
    # print(mum.shape)
    # mu2 = np.mean(Xk2, axis=1)
    # print(mu2.shape)
    # print(np.all(mu==mu2))
    # # print(mu)
    # # print(mum)
    # print(mu2)




    #### center X
    Xc = np.asarray([np.subtract(X[i], mu[c, :]) for c in classes for i in range(dataNo) if labels[i] == c])

    #### center Xk sorted by class
    # Xkc = np.asarray([np.subtract(Xk[c, i, :], mu[c, :]) for c in classes for i in range(Xk.shape[1])])

    #### sort Xc centered by category
    Xck = np.asarray([[Xc[i] for i in range(dataNo) if labels[i] == clas] \
                      for clas in classes])

    s = np.zeros((dims, dims))

    for c in range(Xck.shape[0]):
        totweight = 0
        for i in range(dataNo):
            if (labels[i] == classes[c]):
                s = np.add(s, np.outer(Xc[i, :], Xc[i, :])* W[i])
                totweight += W[i]
                # print(count)
        s /= totweight;
        sigma[:, :, c] = s
        # sigma[:,:,c]

    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 vector of class priors
#         mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
# out:     h - N x 1 class predictions for test points
def classify(X,prior,mu,sigma,covdiag=True):
    # Your code here
    # Example code for solving a psd system
    # L = np.linalg.cholesky(A)
    # y = np.linalg.solve(L,b)
    # x = np.linalg.solve(L.H,y)
    # Your code here
    # Example code for solving a psd system
    # priors = computePrior(labels)
    logPosteriors = np.zeros((sigma.shape[2], X.shape[0]))
    classesNo =sigma.shape[2]
    dims = X.shape[1]
    dataNo = X.shape[0]
    # print(X.shape)
    # print(mu.shape)
    # print(sigma.shape)
    # print(logPosteriors.shape)
    # print(classesNo)
    # print(dataNo)
    # print(dims)
    for cls in range(classesNo):

        # print(sigmac.shape)
        for i in range(dataNo):
            if is_symmetric(sigma[:, :, cls]) and is_positive_definite(sigma[:, :, cls]):
                if covdiag:
                    sigmac = np.diag(np.diag(sigma[:, :, cls]))
                    b = np.subtract(X[i], mu[cls])
                    try:
                        sigmaInv = np.linalg.inv(sigmac)
                        L = sigmaInv
                        # print(sigmac)
                        # print(sigmaInv)
                        # print("Inverse found!!")
                    except np.linalg.LinAlgError:
                        # Not invertible. Skip this one.
                        print("No inverse")
                        # sigmac = sigma[:, :, cls]
                        # A = sigma[:,:,cls]
                        # L = np.linalg.cholesky(A)
                        # y = np.linalg.solve(L, b)
                        # # x = np.linalg.solve(L.getH(), y)
                        # x = np.linalg.solve(np.transpose(L), y)

                    #         else:
                    # continue with what you were doing

                    lnl = np.log(np.linalg.det(sigmac))
                    logPosteriors[cls, i] = -1 / 2 * lnl \
                                            - 1 / 2 * np.dot(np.dot(b, L), np.transpose(b)) \
                                            + np.log(prior[cls])
                else:
                    # L, y = computeY(cls, X[i], mu[cls], sigma[:, :, cls], covdiag)
                    sigmac = sigma[:, :, cls]
                    L = np.zeros(sigmac.shape)
                    x = 0
                    A = sigmac
                    b = np.transpose(np.subtract(X[i], mu[cls]))
                    L = np.linalg.cholesky(A)
                    y = np.linalg.solve(L, b)
                    # x = np.linalg.solve(L.getH(),y)
                    x = np.linalg.solve(np.transpose(L), y)


                    # lnl= np.log(np.linalg.det(sigmac))
                    lnl = 2 * np.sum(np.log(np.diag(L)))

                    # return L, np.transpose(x)

                    # print(.shape)
                    # print(.shape)
                    # print(np.transpose(b).shape)
                    # print( x.shape)
                    # print( np.dot(np.transpose(b), x).shape)
                    # print((-1 / 2 * lnl \
                    #        - 1 / 2 * np.dot(np.transpose(b), x) \
                    #        + np.log(prior[cls])).shape)
                    # logPosteriors[cls, i] = -1 / 2 * lnl \
                    #                         - 1 / 2 * np.dot(b, np.transpose(x)) \
                    #                         + np.log(prior[cls])

                    # logPosteriors[cls, i] = -1 / 2 * lnl \
                    #                         - 1 / 2 * np.dot(b, x) \
                    #                         + np.log(prior[cls])

                    logPosteriors[cls, i] = -1 / 2 * lnl \
                                            - 1 / 2 * np.dot(np.transpose(b), x) \
                                            + np.log(prior[cls])
                    # logPosteriors[cls, i] = -1 / 2 * lnl \
                    #                         - 1 / 2 * np.dot(np.transpose(b), np.transpose(x)) \
                    #                         + np.log(prior)

    # print(logPosteriors)
    h = np.argmax(logPosteriors, axis=0)
    # print(logPosteriors.shape)
    return h


# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.

# X, labels = genBlobs(centers=5)
# # W = 1/X.shape[0] * np.ones(X.shape[0])
# mu, sigma = mlParams(X,labels)
# plotGaussian(X,labels,mu,sigma)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.

# in:       X - N x d matrix of N data points
#      labels - N x 1 vector of class labels
#           T - number of boosting iterations
# out: priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
def trainBoost(X,labels,T=5,covdiag=True):
    # Your code here
    # return priors,mus,sigmas,alphas
    c = len(set(labels))
    d = len(X[0])
    n = len(labels)

    priors = np.zeros([T, c])
    mus = np.zeros([T, c, d])
    sigmas = np.zeros([T, d, d, c])
    alphas = np.zeros(T)

    weights = np.ones(n) / n
    # print('weights', weights)
    for t in range(T):
        mu, sigma = mlParams(X, labels, weights)
        prior = computePrior(labels, weights)
        hi = classify(X, prior, mu, sigma, covdiag)
        priors[t] = prior
        mus[t] = mu
        sigmas[t] = sigma

        error_sum = 0
        for label_index, label in enumerate(labels):
            if hi[label_index] != label:
                error_sum += weights[label_index]

        # print('error_sum', error_sum)
        if error_sum == 0:
            error_sum = 0.0000001

        alphas[t] = (np.log(1 - error_sum) - np.log(error_sum)) / 2

        for label_index, label in enumerate(labels):
            if hi[label_index] == label:
                weights[label_index] *= np.exp(-alphas[t])
            else:
                weights[label_index] *= np.exp(alphas[t])

        sum = 0
        for weight in weights:
            sum += weight
        weights /= sum

    return priors, mus, sigmas, alphas
    pass

# in:       X - N x d matrix of N data points
#      priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
# out:  yPred - N x 1 class predictions for test points
def classifyBoost(X,priors,mus,sigmas,alphas,covdiag=True):
    # Your code here
    # return c
    n = len(X)
    T = len(alphas)
    c = len(priors[0])
    matrix = np.zeros([n, c])
    for t in range(T):
        ht = classify(X, priors[t], mus[t], sigmas[t], covdiag)
        for ni in range(n):
            matrix[ni][ht[ni]] += alphas[t]

    yPred = np.empty(n)
    for ni in range(n):
        likliest_class = None
        highest_vote_value = None
        for ci in range(c):
            if likliest_class is None or matrix[ni][ci] > highest_vote_value:
                likliest_class = ci
                highest_vote_value = matrix[ni][ci]
        yPred[ni] = likliest_class
    return yPred


# ## Define our testing function
#
# The function below, `testClassifier`, will be used to try out the different datasets. `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`. Observe that we split the data into a **training** and a **testing** set.

np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=25)
np.set_printoptions(linewidth=200)

def testClassifier(dataset='iris',dim=0,split=0.7,doboost=False,boostiter=5,covdiag=True,ntrials=100):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        # xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplit(X,y,split)
        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim
        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        ## Boosting
        if doboost:
            # Compute params
            priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter)
            yPr = classifyBoost(xTe,priors,mus,sigmas,alphas,covdiag=covdiag)
        else:
        ## Simple
            # Compute params
            prior = computePrior(yTr)
            mu, sigma = mlParams(xTr,yTr)
            # Predict
            yPr = classify(xTe,prior,mu,sigma,covdiag=covdiag)

        # Compute classification error
        print("Trial:",trial,"Accuracy",100*np.mean((yPr==yTe).astype(float)))

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", np.mean(means), "with standard deviation", np.std(means))
    with open('results.txt', 'a') as f:
        f.write('dataset: ' + dataset + ', covdiag: ' + str(covdiag) + ', boost: ' + str(doboost) + '\n')
        f.write('Final mean classification accuracy: ' + str(np.mean(means)) + ' ' + 'Standard deviation: ' + str(np.std(means)) + '\n')
        f.write('\n')


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.

def plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)
    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)
    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    if doboost:
        ## Boosting
        # Compute params
        priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter,covdiag=covdiag)
    else:
        ## Simple
        # Compute params
        prior = computePrior(yTr)
        mu, sigma = mlParams(xTr,yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            if doboost:
                ## Boosting
                grid[yi,xi] = classifyBoost(np.matrix([[xx, yy]]),priors,mus,sigmas,alphas,covdiag=covdiag)
            else:
                ## Simple
                grid[yi,xi] = classify(np.matrix([[xx, yy]]),prior,mu,sigma,covdiag=covdiag)

    classes = range(np.min(y), np.max(y)+1)
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass   
        xc = pX[py == c, :]
        plt.scatter(xc[:,0],xc[:,1],marker='o',c=color,s=40,alpha=0.5)

    plt.xlim(np.min(pX[:,0]),np.max(pX[:,0]))
    plt.ylim(np.min(pX[:,1]),np.max(pX[:,1]))
    fignam = dataset + '_' + 'covdiag_' + str(covdiag) + '_boost_' + str(doboost) + '.png'
    plt.figure(fignam)
    # plt.draw()
    # plt.show(block=False)

    # plt.savefig(fignam)


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.

# Example usage of the functions
def run(dataset, split=0.7, doboost=False, boostiter=5, covdiag=True):
    testClassifier(dataset=dataset, split=split, doboost=doboost, boostiter=boostiter, covdiag=covdiag)
    plotBoundary(dataset=dataset, split=split, doboost=doboost, boostiter=boostiter, covdiag=covdiag)

def main():
    # testEstimates()
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(precision=25)
    np.set_printoptions(linewidth=200)
    # datasets = ('iris', 'vowel', 'olivetti', 'wine')
    datasets = ('iris', 'vowel', 'olivetti')

    for set in datasets:
        run(set, doboost=False, covdiag=True)
        run(set, doboost=False, covdiag=False)
        run(set, doboost=True, covdiag=True)
        run(set, doboost=True, covdiag=False)

if __name__ == '__main__':
    main()
# testClassifier(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True)
# plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True)
# testClassifier(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=False)
# plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=False)

# #######Bug
# testClassifier(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=True)
# plotBoundary(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=True)
# testClassifier(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=False)
# plotBoundary(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=False)

# testClassifier(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=True)
# plotBoundary(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=True)
# testClassifier(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=False)
# plotBoundary(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=False)

# testClassifier(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=True)
# plotBoundary(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=True)
# testClassifier(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=False)
# plotBoundary(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=False)

# X, labels = genBlobs(centers=5)
# X = np.arange(8).reshape(4,2)
# labels = np.asarray([0,0,1,1])
# print(X.shape[0])
# W = 1/X.shape[0] * np.ones(X.shape[0])
# W = np.ones(X.shape[0])
# W = np.arange(4)
# print(W)
# print(X)
# mu, sigma = mlParams(X,labels, W)

X, labels = genBlobs(centers=5)
W = 1/X.shape[0] * np.ones(X.shape[0])
# W = np.ones(X.shape[0])
mu, sigma = mlParams(X,labels, W)
plotGaussian(X,labels,mu,sigma)