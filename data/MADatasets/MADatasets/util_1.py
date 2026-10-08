# Copyright (c) 2020 Julian Gil Gonzalez
# Universidad Tecnologica de Pereira and University of Sheffield


import random
import warnings
import numpy as np
import climin
from functools import partial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#Defining the Sigmoid function and Softmax function
def Sigmoid(f_r):
    lam_r = 1/(1 + np.exp(-f_r))
    return lam_r

def MAjVot(Y, K):
    N,R = Y.shape
    Yhat = np.zeros((N,1))
    for n in range(N):
        votes = np.zeros((K,1))
        for r in range(R):
            for k in range(K):
                if Y[n,r] == k+1:
                    votes[k] = votes[k]+1
        Yhat[n] = np.argmax(votes) + 1
    return Yhat

# Genración de etiquetas desde múltiples anotadores. Asume anotadores cuyo rendimiento es una
# función del espacio de entrada.
# Fija NrP = 1.
def  MA_Clas_Gen(Xtrain,ytrain,R,NrP):
    
    P = np.asarray([
        [0.00, 0.90, 0.50, 0.15, 0.60],
        [0.90, 0.00, 0.30, 0.40, 0.75],
        [0.50, 0.30, 0.00, 0.60, 0.30],
        [0.15, 0.40, 0.60, 0.00, 0.80],
        [0.60, 0.75, 0.30, 0.80, 0.00],
        ])
    
    kmeans = KMeans(n_clusters=R, random_state=0).fit(Xtrain)
    K = len(np.unique(ytrain))
    N = len(ytrain)
    Ytrain = np.zeros((N, R))
    for r in range(R):
        aux = ytrain.copy()
        for c in range(R):
            auxidx = np.argwhere(kmeans.labels_ == c)
            Nk = len(auxidx)
            a = np.random.binomial(1, P[c,r], Nk)
            for i in range(Nk):
                if a[i] != 1:
                    Ytrain[auxidx[i],r] = ytrain[auxidx[i]]
                else:
                    auxlab = np.random.permutation(K) + 1
                    auxlab = np.delete(auxlab,(auxlab==ytrain[auxidx[i]]).flatten())
                    Ytrain[auxidx[i],r] = auxlab[0]
                    
                
                
        
    iAnn = np.zeros((N, R), dtype=int) # this indicates if the annotator r labels the nth sample.
    Nr = np.ones((R), dtype=int)*int(np.floor(N*NrP))  
    for r in range(R):
        if r < R-1:
            indexR = np.random.permutation(range(N))[:Nr[r]]
            iAnn[indexR,r] = 1
        else:
            iSimm = np.sum(iAnn, axis=1)
            idxZero = np.asarray([i for (i, val) in enumerate(iSimm) if val == 0])
            Nzeros = idxZero.shape[0]
            idx2Choose = np.arange(N)
            if Nzeros == 0:
                indexR = np.random.permutation(range(N))[:Nr[r]]
                iAnn[indexR,r] = 1
            else:
                idx2Choose = np.delete(idx2Choose, idxZero)
                N2chose = idx2Choose.shape[0]
                idxNoZero = np.random.permutation(N2chose)[:(Nr[r] - Nzeros)]
                idxTot = np.concatenate((idxZero, idx2Choose[idxNoZero]))
                iAnn[idxTot,r] = 1
    
    # Now, we verify that all the samples were labeled at least once
    Nr = (np.sum(iAnn,0))
    iSimm = np.sum(iAnn, axis=1)
    if np.asarray([i for (i, val) in enumerate(iSimm) if val == 0]).sum() == 0:
        ValueError("all the samples must be labeled at least once")

    # Finally, if iAnn=0 we assign a reference value to indicate a missing value
    Vref = -1e-20
    for r in range(R):
        Ytrain[iAnn[:,r] == 0, r] = Vref 

    return Ytrain, iAnn

def CrossVal(X, pp, Nk):
    N = X.shape[0]
    Ntr = int(N*pp)
    Nte = N - Ntr
    idxtr = np.zeros((Ntr,Nk))
    idxte = np.zeros((Nte,Nk))
    
    for i in range(Nk):
        index = np.random.permutation(range(N))
        idxtr[:,i] = index[:Ntr]
        idxte[:,i] = index[Ntr:]
        
    return idxtr, idxte
    
    
        
