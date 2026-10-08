import pandas as pd
import numpy as np
import util as ut
import scipy.io as sio
import os
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_squared_log_error
Dir = os.path.dirname(os.path.realpath('__file__'))

#Some useful variables
Nk = 30 # number iteration for the cross validation.
pp = 0.7 # percentage of samples used for the training set.
R = 5 #number of annotators.
Nrp = 1.0 #number of samples labeled by each annotator (perecentage).


# name = "Breast"
# filename = "breast-cancer-wisconsin1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()+1
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))


# name = "Bupa"
# filename = "bupa1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()+1
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))

# name = "Ionosphere"
# filename = "ionosphere1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()+1
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))


# name = "Iris"
# filename = "iris1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))

# name = "Pima"
# filename = "pima-indians-diabetes1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()+1
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))


# name = "Segmentation"
# filename = "segmentation1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))


# name = "TicTacToe"
# filename = "tic-tac-toe1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()+1
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))


# name = "Wine"
# filename = "wine1.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['MAData']['X'].item()
# y = D['MAData']['t'].item()
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))

# name = "Western"
# filename = "Western.mat"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = sio.loadmat(filenam, squeeze_me=True)
# X = D['X']
# y = D['y']
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))

# name = "Skin_NonSkin"
# filename = "Skin_NonSkin.txt"
# fileDir = '../Dataset/'+filename
# filenam = os.path.join(Dir, fileDir)
# filenam = os.path.abspath(os.path.realpath(filenam))
# D = pd.read_csv(filenam, header = None, sep='\t')
# X = D.iloc[:,:3].to_numpy()
# y = D.iloc[:,3].to_numpy()
# Nt = len(y)
# y = y.reshape((Nt,1))

# K = len(np.unique(y))
# Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
# Lam_r = np.asarray(Lam_r).T
# Lam_r = np.squeeze(Lam_r, axis=0)
# idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
# sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
# Ymv = ut.MAjVot(Y, K)
# print(accuracy_score(y, Y[:,0]))
# print(accuracy_score(y, Y[:,1]))
# print(accuracy_score(y, Y[:,2]))
# print(accuracy_score(y, Y[:,3]))
# print(accuracy_score(y, Y[:,4]))
# print(accuracy_score(y, Ymv))

name = "Occupancy"
Number = ['1', '2', '3']
for i in Number:
    filename = "Occupancy" + i + ".txt"
    fileDir = '../Dataset/'+filename
    filenam = os.path.join(Dir, fileDir)
    filenam = os.path.abspath(os.path.realpath(filenam))
    D = pd.read_csv(filenam, header = None,skiprows=[0])
    
    if int(i) == 1:
        X = D.iloc[:,2:7].to_numpy()
        y = D.iloc[:,7].to_numpy() + 1
    else:
        X = np.concatenate((X, D.iloc[:,2:7].to_numpy()), axis=0)
        y = np.concatenate((y, D.iloc[:,7].to_numpy()+1), axis=0)
    
Nt = len(y)
y = y.reshape((Nt,1))

K = len(np.unique(y))
Y, iAnn, Lam_r = ut.MA_Clas_Gen(X,y,R,Nrp) #Generating data from multiple annotators.
Lam_r = np.asarray(Lam_r).T
Lam_r = np.squeeze(Lam_r, axis=0)
idxtr, idxte = ut.CrossVal(X, pp, Nk)  #Generating index for the crossvalidation.
sio.savemat(name+'.mat', {'X': X, 'y':y, 'Y':Y, 'iAnn':iAnn, 'Exp':Lam_r, 'idxtr':idxtr, 'idxte':idxte})
Ymv = ut.MAjVot(Y, K)
print(accuracy_score(y, Y[:,0]))
print(accuracy_score(y, Y[:,1]))
print(accuracy_score(y, Y[:,2]))
print(accuracy_score(y, Y[:,3]))
print(accuracy_score(y, Y[:,4]))
print(accuracy_score(y, Ymv))

