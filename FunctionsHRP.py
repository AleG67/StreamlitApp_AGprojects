####### HIERARCHICAL RISK PARITY #######

import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd

def getIVP(cov, **kargs):
    """
    Compute the inverse-variance portfolio using a given formula for the weights
     - cov = covariance matrix
    """
    ivp = 1. / np.diag(cov) 
    ivp /= ivp.sum() 
    return ivp

def getClusterVar(cov,cItems):
    """
    Compute variance for the cluster by slicing the covariance matrix given the items to cluster.
    Uses the function getIVP to obtain weigths for the clustered sub-covmatrix.
     - cov = entire covariance matrix
     - cItems = list or array of labels to obtain the clustered sub-covmatrix
    """
    cov_ = cov.loc[cItems, cItems]    # matrix slice given the items to cluster together
    w_ = getIVP(cov_).reshape(-1,1) 
    cVar = np.dot(np.dot(w_.T,cov_), w_)[0,0] 
    return cVar

def getQuasiDiag(link):
    """
    Quasi-diagonaliazation procedure to sort clustered items by distance, 
    putting the objects that are more connected closer together.
     - link = output of clustering from sch.linkage(dist, 'single') 
            with dist = correlation matrix output of the function correlDist
    """
    link = link.astype(int) 
    sortIx = pd.Series([link[-1,0],link[-1,1]])  
    numItems = link[-1,3]    # number of original items 
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0]*2, 2)  # make space 
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index 
        j = (df0.values - numItems)
        sortIx[i] = link[j,0] # item 1 
        df0 = pd.Series(link[j,1],index=i+1) 
        sortIx = sortIx.append(df0)  # item 2 
        sortIx = sortIx.sort_index() # re-sort 
        sortIx.index = range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

def getRecBipart(cov,sortIx):
    """
    Compute HRP allocation (weights) using the Recursive Bisection algorithm.
    Use function getClusterVar defined above.
     - cov = covariance matrix of returns
     - sortIX = corr.index[SORTIX].tolist() with SORTIX output of the function getQuasiDiag and corr = correlation matrix of returns
    """
    w = pd.Series(1,index=sortIx)
    cItems = [sortIx] # initialize all items in one cluster 
    while len(cItems)>0:
        cItems = [i[j:k] for i in cItems for j,k in ((0,len(i)//2), (len(i)//2, len(i))) if len(i)>1] # bi-section
        for i in range(0, len(cItems), 2): 
            # parse in pairs 
            cItems0 = cItems[i]   # cluster 1 
            cItems1 = cItems[i+1] # cluster 2 
            cVar0 = getClusterVar(cov, cItems0) 
            cVar1 = getClusterVar(cov, cItems1) 
            alpha = 1 - (cVar0/(cVar0+cVar1))
            w[cItems0] *= alpha   # weight 1 
            w[cItems1] *= 1-alpha # weight 2
    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1      
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5  # distance matrix
    return dist

def getHRP(cov,corr):
    """
    Uses the function defined above to Construct a hierarchical portfolio
     - cov = covariance matrix of returns
     - corr = correlation matrix of returns
    The OUTPUT is a list of weights.
    """
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov) 
    dist = correlDist(corr) 
    link = sch.linkage(dist,'single') 
    sortIx = getQuasiDiag(link) 
    sortIx = corr.index[sortIx].tolist() # recover labels 
    hrp = getRecBipart(cov,sortIx)
    return hrp.sort_index()