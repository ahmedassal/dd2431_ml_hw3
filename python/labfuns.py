import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rnd
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.patches import Ellipse


def genData():
    # Generate random Gaussian data 
    X1 = np.array([(rnd.normalvariate(1.0, 0.5), rnd.normalvariate(1.0, 0.5)) for i in range(100)])
    X2 = np.array([(rnd.normalvariate(0.0, 0.5), rnd.normalvariate(5.0, 0.5)) for i in range(100)])
    X3 = np.array([(rnd.normalvariate(3.0, 0.5), rnd.normalvariate(5.0, 0.5)) for i in range(100)])
    y1 = np.ones((100,))
    y2 = 2 * np.ones((100,))
    y3 = np.zeros((100,))
    X = np.vstack((X1,X2,X3))
    y = np.hstack((y1,y2,y3))
    return X,y



def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.25)

    ax.add_artist(ellip)
    return ellip



# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data.
def trteSplit(X,y,pcSplit):
    # Compute split indices
    Ndata = X.shape[0]
    Ntr = np.rint(Ndata*pcSplit)
    Nte = Ndata-Ntr
    idx = np.random.permutation(Ndata)
    trIdx = idx[:Ntr]
    teIdx = idx[Ntr:]
    # Split data
    xTr = X[trIdx,:]
    yTr = y[trIdx]
    xTe = X[teIdx,:]
    yTe = y[teIdx]
    return xTr,yTr,xTe,yTe,trIdx,teIdx


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to  
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTr = np.zeros((0,))
    yTe = np.zeros((0,))
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        yclass = y[classIdx]
        NPerClass = np.sum(classIdx)
        Ntr = np.rint(NPerClass*pcSplit)
        idx = np.random.permutation(NPerClass)
        trIdx = idx[:Ntr]
        teIdx = idx[Ntr:]
        # Split data
        xTr = np.vstack((xTr,Xclass[trIdx,:]))
        yTr = np.hstack((yTr,yclass[trIdx]))
        xTe = np.vstack((xTe,Xclass[teIdx,:]))
        yTe = np.hstack((yTe,yclass[teIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx    


def fetchDataset(dataset='iris'):
    if dataset == 'iris':
        X = genfromtxt('irisX.txt', delimiter=',')
        y = genfromtxt('irisY.txt', delimiter=',',dtype=np.int)-1
        pcadim = 0
    elif dataset == 'wine':
        X = genfromtxt('wineX.txt', delimiter=',')
        y = genfromtxt('wineY.txt', delimiter=',',dtype=np.int)-1 
        pcadim = 0  
    elif dataset == 'olivetti':
        X = genfromtxt('olivettifacesX.txt', delimiter=',')
        X = X/255
        y = genfromtxt('olivettifacesY.txt', delimiter=',',dtype=np.int)
        pcadim = 5
    elif dataset == 'vowel':
        X = genfromtxt('vowelX.txt', delimiter=',')
        y = genfromtxt('vowelY.txt', delimiter=',',dtype=np.int)
        pcadim = 0
    else:
        print "Please specify a dataset!"
        X = np.zeros(0)
        y = np.zeros(0)
        pcadim = 0
        
    return X,y,pcadim



def genBlobs(n_samples=200,centers=5,n_features=2):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=0)
    return X,y


# Scatter plots the two first dimension of the given data matrix X
# and colors the points by the labels.
def scatter2D(X,y):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()



def plotGaussian(X,y,mu,sigma):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        plot_cov_ellipse(sigma[:,:,label], mu[label])
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()
