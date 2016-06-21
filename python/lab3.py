
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
        sumW = np.sum(W,axis = 0)
        prior = np.zeros(classesNo)
        tot = 0
        for cls in range(classesNo):
            for i in range(dataNo):
                if(labels[i]==cls):
                    prior[classes[cls]] += W[i]
                    tot += W[i]
        prior[:] /= tot

    return prior

# Note that you do not need to handle the W argument for this part
# in:      X - N x d matrix of N data points
#     labels - N x 1 vector of class labels
#     W      - N x 1 vector of weights
# out:    mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
def mlParams(X,labels,W=None):
    # Your code here
    # print("X ", X.shape)
    # print("X ", X)
    # print("W ", W.shape)
    # print("W ", W)
    labels = [int(label) for label in labels]
    classes = np.unique(labels)
    classesNo = classes.shape[0]
    dims = X.shape[1]
    dataNo = X.shape[0]
    if W is None:
        W = np.ones(dataNo)

    mu = np.zeros((classesNo, dims));
    sigma = np.zeros((dims, dims, classesNo))

    #### sort X by class
    Xt = np.transpose(X)
    Xw = np.transpose(np.multiply( W, Xt))

    # print("Xw ", Xw.shape)
    # print("Xw ", Xw)
    # print("dataNo ", dataNo)

    Xk = np.asarray([[Xw[i] for i in range(dataNo) if labels[i] == clas] \
                     for clas in classes])

    # print("Xk ", Xk.shape)
    # Xk2 = np.asarray([[X[i] for i in range(dataNo) if labels[i] == clas] \
    #                  for clas in classes])
    # Xk2 = []
    #
    # for clas in classes:
    #     classEl=[]
    #     for i in range(dataNo):
    #         if (labels[i] == clas) :
    #             classEl.append(X[i])
    #     classEl = np.asarray(classEl)
    #     print("classEl ", classEl.shape)
    #     Xk2.append(classEl)
    # Xk2 = np.asarray(Xk2)
    # print("Xk ", Xk.shape)
    #
    # print("Xk2 ", Xk2.ndim)


    ## New way
    muk = np.zeros((dims))

    # Wk = np.zeros((dims))
    # print("classesNo", classesNo)
    Wkss = []
    mu = []
    for cls in range(classesNo):
        classEl = []
        # Wk = []
        sumWk = 0
        for i in range(dataNo):
            if (labels[i] == cls):
                classEl.append(Xw[i])
                sumWk +=  W[i]
                # Wk.append(W[i])
        classEl = np.asarray(classEl)
        # Wk = np.asarray(Wk)
        # print("Wk ", Wk)
        # Wks = [np.sum(Wk, axis=0)]
        # print("Wks ", Wks)
        # Wkss.append(Wks)
        # muk = np.multiply(Wk, classEl)
        # print("muk ", muk)
        # muk = np.sum(np.asarray(muk), axis = 0) / Wks[0]
        muk = np.sum(classEl, axis=0) / sumWk
        # print("muk ", muk)
        mu.append(muk)
    # Wkss = np.asarray(Wkss)
    # print("Wkss ", Wkss.shape)
    # print("Wkss ", Wkss)
    mu = np.asarray(mu)
    # print("mu ", mu.shape)
    # print("mu ", mu)
    ## old
    # Wk = np.asarray([[W[i] for i in range(dataNo) if labels[i] == clas] \
    #                  for clas in classes])
    # for cls in range(classesNo):
    #     print("cls ", cls, " ", classesNo)
    #     for d in range(dims):
    #         print("dims ", d, " ", dims)
    #         print("Xk2 ", Xk2.shape)
    #         for n in range(Xk2.shape[1]):
    #             print("Xk2 ", n, " ", Xk2.shape[1])
    #             print(cls, d, n)
    #             mu[cls, d] += Xk2[cls, n, d] * Wk[cls, n]  # (mu[cls, d] / Wks[cls]

    # Wks = np.sum(Wk, axis=1)
    # for cls in range(classesNo):
    #     for d in range(mu.shape[1]):
    #         mu[cls, d] = mu[cls, d] / Wks[cls]

    #### center X
    # Xc = np.asarray([np.subtract(X[i], mu[c, :]) for c in classes for i in range(dataNo) if labels[i] == c])

    #### sort Xc centered by class
    # Xck = np.asarray([[Xc[i] for i in range(dataNo) if labels[i] == clas] \
    #                   for clas in classes])
    # print("X ", X[:])
    # print("mu ", mu[:,:])


    for cls in range(classesNo):
        s = np.zeros((dims, dims))
        totweight = 0
        # print("cls ", cls)
        for i in range(dataNo):
            if (labels[i] == classes[cls]):
                Xc = np.subtract(X[i, :], mu[cls, :])
                # s = np.add(s, np.outer(Xc[i, :], Xc[i, :])* W[i])
                # print("X ", X[i])
                # print("mu ", mu[cls,:])
                # print("Xc ", Xc)
                # print("outer(Xc[:], Xc[:])", np.outer(Xc[:], Xc[:]))

                # print("W ", W[i])
                # print("outer(Xc[:], Xc[:]) * W[i]", np.outer(Xc[:], Xc[:])* W[i])

                s = np.add(s, np.outer(Xc[:], Xc[:]) * W[i])
                # print("S ", s)
                totweight += W[i]
                # print("totweight ", totweight)
        s /= totweight
        # print("S ", s)
        sigma[:, :, cls] = s
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
    classesNo = sigma.shape[2]
    dims = X.shape[1]
    dataNo = X.shape[0]

    logPosteriors = np.zeros((classesNo, dataNo))

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

                    # else:
                    # continue with what you were doing
                    lnl = np.log(np.power(np.linalg.det(sigmac),2))
                    logPosteriors[cls, i] = -1 / 2 * lnl \
                                            - 1 / 2 * np.dot(np.dot(b, L), np.transpose(b)) \
                                            + np.log(prior[cls])
                else:
                    # L, y = computeY(cls, X[i], mu[cls], sigma[:, :, cls], covdiag)
                    sigmac = sigma[:, :, cls]
                    L = np.zeros(sigmac.shape)
                    x = 0
                    A = sigmac
                    # b = np.transpose(np.subtract(X[i], mu[cls]))
                    b = np.subtract(X[i], mu[cls])
                    L = np.linalg.cholesky(A)
                    v = np.linalg.solve(L, np.transpose(b))
                    # x = np.linalg.solve(L.getH(),y)
                    y = np.transpose([np.linalg.solve(np.transpose(L), v)])


                    lnl= np.log(np.linalg.det(sigmac))

                    # print("Full ", np.dot(np.dot(b, y), np.transpose(b)))
                    # print("Full ", np.dot(b, v))

                    # t1 = np.dot(b, y)
                    # t2 = np.dot(y, np.transpose(b))
                    # b = b.reshape((4,1))
                    # y = y.reshape((4, 1))
                    # print("y ", y, " ", y.shape)
                    # print("b ", b, " ", b.shape)
                    # print("np.transpose(np.transpose(b)) ", np.transpose(np.transpose(b)))
                    # print("np.transpose(np.transpose(b)) ", np.transpose(np.transpose(b)).shape)
                    # print("y ", y)
                    # print("y ", y.shape)
                    logPosteriors[cls, i] = -1 / 2 * lnl \
                                            - 1 / 2 * np.dot(np.transpose(np.transpose(b)), y)  \
                                            + np.log(prior[cls])

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

    # data stats
    # c = len(set(labels))
    # d = len(X[0])
    # n = len(labels)
    classes = np.unique(labels)
    classesNo = classes.shape[0]
    dims = X.shape[1]
    dataNo = X.shape[0]

    # data sturcts initializations
    # priors = np.zeros((T, c))
    # mus = np.zeros((T, c, d))
    # sigmas = np.zeros([T, d, d, c])
    # alphas = np.zeros(T)
    priors = np.zeros((T, classesNo))
    mus = np.zeros((T, classesNo, dims))
    sigmas = np.zeros((T, dims, dims, classesNo))
    alphas = np.zeros(T)

    # Step 0  - weights initialization
    weights = np.ones(dataNo) / dataNo
    # print('weights', weights)

    for t in range(T):
        # Step 1 - train weak learner - compute parameters of the gaussians of each class
        mu, sigma = mlParams(X, labels, weights)
        prior = computePrior(labels, weights)
        # Step 2 - get weak hypothesis hi
        hi = classify(X, prior, mu, sigma, covdiag)

        # save the data for the iteration round t of T
        priors[t] = prior
        mus[t] = mu
        sigmas[t] = sigma

        # Step 3 - compute alpha
        # compute sum of errors times weights
        epsilon = 0
        for idx, label in enumerate(labels):
            if hi[idx] != label:
                epsilon += weights[idx]

        # print('error_sum', error_sum)
        if epsilon == 0:
            epsilon = 0.0000001

        alphas[t] = 1/2 * (np.log(1 - epsilon) - np.log(epsilon))

        # Step 4 - update weights
        for idx, label in enumerate(labels):
            if hi[idx] == label:
                weights[idx] *= np.exp(-alphas[t])
            else:
                weights[idx] *= np.exp(alphas[t])

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
    # n = len(X)
    # T = len(alphas)
    # c = len(priors[0])
    # T = alphas.shape[0]

    classesNo = priors.shape[1]
    dims = X.shape[1]
    dataNo = X.shape[0]
    T = alphas.shape[0]

    # print("T: ", T, "dataNo: ", dataNo, "classesNo: ", classesNo)
    # print(priors.shape)

    votes = np.zeros((dataNo, classesNo))
    for t in range(T):
        hypo = classify(X, priors[t], mus[t], sigmas[t], covdiag)
        for i in range(dataNo):
            # print("T: ", t, " votes[",i, "][",hypo[i],"] += alphas[", t, "]", " = ",  alphas[t])
            votes[i][hypo[i]] += alphas[t]

    H = np.empty(dataNo)
    # for i in range(dataNo):
    #     likliest_class = None
    #     highest_vote_value = None
    #     for cls in range(c):
    #         if likliest_class is None or votes[i][cls] > highest_vote_value:
    #             likliest_class = cls
    #             highest_vote_value = votes[i][cls]
    #     H[i] = likliest_class
    H = np.argmax(votes, axis=1)
    # print("H ", H)
    return H


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
    figtitle = dataset + '_' + 'covdiag_' + str(covdiag) + '_boost_' + str(doboost)
    fignam = figtitle + '.png'
    plt.figure(fignam)
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

    # plt.hold(False)
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

    # fig = plt.figure(1)
    # plt.draw()
    plt.title(figtitle)
    # plt.show(block=False)
    # plt.show(block=True)

    plt.savefig(fignam)
    # fig.savefig(fignam, dpi=fig.dpi)
    # plt.close(fig)


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.

# Example usage of the functions
def run(dataset, split=0.7, doboost=False, boostiter=5, covdiag=True):
    testClassifier(dataset=dataset, split=split, doboost=doboost, boostiter=boostiter, covdiag=covdiag)
    plotBoundary(dataset=dataset, split=split, doboost=doboost, boostiter=boostiter, covdiag=covdiag)

def main1():
    X, labels = genBlobs(centers=5)
    # W = 1 / X.shape[0] * np.ones(X.shape[0])
    # W = np.ones(X.shape[0])
    mu, sigma = mlParams(X, labels)
    plotGaussian(X, labels, mu, sigma)

def main_testclassifier():
    # testClassifier(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True)
    # plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True)
    testClassifier(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=False)
    # plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=False)

    # testClassifier(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=True)
    # plotBoundary(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=True)
    testClassifier(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=False)
    # plotBoundary(dataset='wine',split=0.7,doboost=False,boostiter=5,covdiag=False)
    #
    # testClassifier(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=True)
    # plotBoundary(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=True)
    testClassifier(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=False)
    # plotBoundary(dataset='vowel',split=0.7,doboost=False,boostiter=5,covdiag=False)
    #
    # testClassifier(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=True)
    # plotBoundary(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=True)
    testClassifier(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=False)
    # plotBoundary(dataset='olivetti',split=0.7,doboost=False,boostiter=5,covdiag=False)
    pass

def main_testboostclassifier():
    testClassifier(dataset='iris',split=0.7,doboost=True,boostiter=5,covdiag=True)
    plotBoundary(dataset='iris',split=0.7,doboost=True,boostiter=5,covdiag=True)
    testClassifier(dataset='iris',split=0.7,doboost=True,boostiter=5,covdiag=False)
    plotBoundary(dataset='iris',split=0.7,doboost=True,boostiter=5,covdiag=False)

    testClassifier(dataset='wine',split=0.7,doboost=True,boostiter=5,covdiag=True)
    plotBoundary(dataset='wine',split=0.7,doboost=True,boostiter=5,covdiag=True)
    testClassifier(dataset='wine',split=0.7,doboost=True,boostiter=5,covdiag=False)
    plotBoundary(dataset='wine',split=0.7,doboost=True,boostiter=5,covdiag=False)

    testClassifier(dataset='vowel',split=0.7,doboost=True,boostiter=5,covdiag=True)
    plotBoundary(dataset='vowel',split=0.7,doboost=True,boostiter=5,covdiag=True)
    testClassifier(dataset='vowel',split=0.7,doboost=True,boostiter=5,covdiag=False)
    plotBoundary(dataset='vowel',split=0.7,doboost=True,boostiter=5,covdiag=False)

    testClassifier(dataset='olivetti',split=0.7,doboost=True,boostiter=5,covdiag=True)
    plotBoundary(dataset='olivetti',split=0.7,doboost=True,boostiter=5,covdiag=True)
    testClassifier(dataset='olivetti',split=0.7,doboost=True,boostiter=5,covdiag=False)
    plotBoundary(dataset='olivetti',split=0.7,doboost=True,boostiter=5,covdiag=False)
    pass

def main_final():
    # testEstimates()
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(precision=25)
    np.set_printoptions(linewidth=200)
    datasets = ('iris', 'vowel', 'olivetti', 'wine')
    # datasets = ('iris', 'vowel', 'olivetti')

    for set in datasets:
        run(set, doboost=False, covdiag=True)
        run(set, doboost=False, covdiag=False)
        run(set, doboost=True, covdiag=True)
        run(set, doboost=True, covdiag=False)


def main_dummydata():
    # X, labels = genBlobs(centers=5)
    X = np.arange(8).reshape(4, 2)
    labels = np.asarray([0, 0, 1, 1])
    # print(X)
    # print(X.shape)
    W = 1 / X.shape[0] * np.ones(X.shape[0])
    # print(W)
    # print(W.shape)
    # W = np.ones(X.shape[0])
    # W = np.arange(4)
    mu, sigma = mlParams(X, labels, W)
    # print("mu ", mu)
    plotGaussian(X, labels, mu, sigma)

def main_priors():
    # X, labels = genBlobs(centers=5)
    X = np.arange(12).reshape(6, 2)
    labels = np.asarray([0, 1, 1, 0, 0, 1])
    # print(X)
    # print(X.shape)
    W = 1 / X.shape[0] * np.ones(X.shape[0])
    priors = computePrior(labels, W)
    print("priors ", priors)

    W = [ 3/2, 1, 1/2, 2, 1/2, 1/2]
    # print(W)
    # print(W.shape)
    # W = np.ones(X.shape[0])
    # W = np.arange(4)
    priors = computePrior(labels, W)
    print("priors ", priors)

def dummyboostdata():
    X = np.arange(48).reshape(24, 2)
    labels = np.asarray([0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1])
    priors, mus, sigmas, alphas = trainBoost(X, labels, T=5, covdiag=True)
    H = classifyBoost(X, priors, mus, sigmas, alphas, covdiag=True)
    SSE = 0
    for i in range(X.shape[0]):
        SSE += np.power((labels[i] - H[i]),1)
        print("X H", labels[i], H[i])
    print("SSE: ", SSE)

def main_instructortests():
    data = [[0.4211, 0.3684],
            [0.3529, 0.3529],
            [0.3000, 0.4000],
            [0.5556, 0.3889],
            [0.5263, 0.3684],
            [0.3250, 0.3500],
            [0.3372, 0.3488],
            [0.3370, 0.3478],
            [0.3759, 0.3534],
            [0.4302, 0.3663],
            [0.4353, 0.3294],
            [0.3594, 0.3438],
            [0.3618, 0.3374],
            [0.3660, 0.3447],
            [0.3632, 0.3498],
            [0.3600, 0.3511],
            [0.3525, 0.3525],
            [0.3534, 0.3534],
            [0.3562, 0.3519],
            [0.3612, 0.3524]]
    data = np.asarray(data)
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    labels = np.asarray(labels)

    mu, sigma = mlParams(data, labels)
    print("mu: ", mu)
    print("sigma: ", sigma)

    prior = computePrior(labels)
    print("prior: ", prior)

    classes = classify(data, prior, mu, sigma, covdiag=False)
    SSE = 0
    for i in range(data.shape[0]):
        SSE += np.power((labels[i] - classes[i]), 2)
        print("X H", labels[i], classes[i])
    print("SSE: ", SSE)
if __name__ == '__main__':
    # main_dummydata()
    # main1()
    # main_testclassifier()
    # main_priors()
    # dummyboostdata()
    # main_instructortests()
    # main_testboostclassifier()
    main_final()











