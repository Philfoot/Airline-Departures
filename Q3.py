
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from scipy.io import loadmat
iv = loadmat('IV.mat')

from pandas import DataFrame
X = DataFrame.from_records(iv["X"])
X1=pd.DataFrame(X[0])
X2=pd.DataFrame(X[1])
X3=pd.DataFrame(X[2])
Y = DataFrame.from_records(iv["Y"])
Z = DataFrame.from_records(iv["Z"])

n=np.array(Y.count())

def g_samp(b):
    return (1/n)*np.matmul(np.transpose(Z),np.subtract(np.subtract(np.subtract(Y,X1*b[0]),X2*b[1]),X3*b[2]))

W = np.identity(4)
def Q(b):
    return np.matmul(np.matmul(np.transpose(g_samp(b)),W),g_samp(b))

from scipy.optimize import basinhopping
x0=np.array([0,0,0])
betas = basinhopping(Q,x0,niter=10)

b_1st = pd.DataFrame(betas.x)
e = Y-np.matmul(X,b_1st)
e2 = np.array(np.square(e))
W_eff = np.linalg.inv(np.matmul(np.transpose(np.multiply(e2,Z)),Z))

def Q2(b):
    return np.matmul(np.matmul(np.transpose(g_samp(b)),W_eff),g_samp(b))

betas2 = basinhopping(Q2,x0,niter=10)


# In[2]:


betas.x


# In[3]:


betas2.x

