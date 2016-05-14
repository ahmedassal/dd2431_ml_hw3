
import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
from sklearn import decomposition
from matplotlib.colors import ColorConverter

X, labels = genBlobs();
#plotGaussianNoEllipses(X, labels)
classes = np.unique(labels)
classesNo = classes.shape[0]
dims = X.shape[1]
dataNo = X.shape[0]
mu = np.zeros((classesNo, dims));
sigma = np.zeros((dims, dims, classesNo))

#### sort X by class
Xk = np.asarray([[X[i] for i in range(dataNo) if labels[i] == clas ] \
                 for clas in classes])
mu = np.mean(Xk, axis = 1)
# print("mu's: ")
# print("------")
# print(mu)

#### center X
Xc = np.asarray([np.subtract(X[i], mu[c, :]) for c in classes for i in range(dataNo) if labels[i] == c ])

#### center Xk sorted by class
Xkc = np.asarray([np.subtract(Xk[c,i,:], mu[c, :]) for c in classes for i in range(Xk.shape[1])])

#### sort Xc centered by category
Xck = np.asarray([[Xc[i] for i in range(dataNo) if labels[i] == clas ] \
                 for clas in classes])

####
#s = np.asarray(np.mean(np.outer(Xck[c,i,:], Xck[c, i, :])   \
#                     for i in range(Xck.shape[1]) for c in classes ], axis = (0)))
s = np.zeros((dims,dims))

for c in range(Xck.shape[0]):
    count = 0
    s = s = np.zeros((dims,dims))
    for i in range(dataNo):
        if(labels[i]==classes[c]):
            #print("there ", count)
            s = np.add(s, np.outer(Xc[i,:], Xc[i, :]))
            #print(s)
            count +=1
            #print(count)
    s/=count;
    sigma[:, :, c] = s
    #print(s)
    #print(sigma[:,:,c])

#print("sigma's: ")
#print("------")
#print(sigma)
#plotGaussian(X,labels,mu,sigma)

# Tests if a matrix A is definite positive
def is_positive_definite(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M+M.transpose()) > 0)

def set_Nan_to_zero(A):
    M = np.matrix(A)
    A[np.isnan(A)] = 0

def is_symmetric(A):
    M = np.matrix(A)
    return np.allclose(A.transpose(1, 0), A)

def convert_to_symmetric_positive_definite(A):
    vals, vecs = np.linalg.eigh(A)
    return A + vecs - 6 * np.identity

def computePrior(labels,W=None):
    # Your code here
    #### sort X by class
    classes = np.unique(labels)
    classesNo = classes.shape[0]
    dataNo = labels.shape[0]
    l = [[labels[i] for i in range(dataNo) if labels[i] == clas ] \
                     for clas in classes]
    prior = np.zeros(len(l))
    prior = [len(l[i])/dataNo for i in range(len(l))]

    return prior

def computeY(cls, x, mu, sigma):
    A = sigma[:,:]
    print("A :")
    print(A.shape)
    print(A)

    b = np.transpose(np.subtract(x, mu))
    print("b :")
    print(b)
    print(b.shape)

    L = np.linalg.cholesky(A)
    print("L :")
    print(L)
    print(type(L))
    print(L.shape)
    print(type(L))

    L = np.matrix(L)

    y = np.linalg.solve(L,b)
    print("y :")
    print(y.shape)
    print(type(y))
    print(y)
    x = np.linalg.solve(L.getH(),y)
    return L, x

def computeLogPosterior(x, cls, prior,mu,sigma, L, y):
    lnl = 2 * np.sum(np.diag(L))
    logPosterior = \
        -1/2 * lnl \
        - 1/2 * np.dot(np.subtract(x, mu), y) \
        + np.log(prior)
    return logPosterior

classes = np.unique(labels)
classesNo = classes.shape[0]
dims = X.shape[1]
dataNo = X.shape[0]

#T = [is_symmetric(sigma[:,:, c]) and is_positive_definite(sigma[:,:, c]) for c in classes]
#print(T)
priors = computePrior(labels)
#sigmaInv = np.zeros(())
logPosteriors = np.zeros((sigma.shape[2], X.shape[0]))

for i in range(X.shape[0]):
    for cls in range(sigma.shape[2]):
        if is_symmetric(sigma[:, :, c]) and is_positive_definite(sigma[:, :, c]):
            L, y = computeY(c, X[i], mu[c], sigma[:, :, c])
            logPosteriors[cls, i] = computeLogPosterior(X[i], cls, \
                                                        priors[cls], \
                                                mu[cls], sigma[:, :, cls], L, y)
            # else:

            #else:



    # values = np.asarray(
    #     [[computeLogPosterior(x, cls, prior[cls], mu[cls], sigma[cls], sigmaInv[cls]) \
    #       for cls in classes] for x in X])
    print("logPosteriors: ")
    print(logPosteriors.shape)
    print(logPosteriors)

    h = np.max(logPosteriors, axis = 0)
    print("h: ")
    print(h)