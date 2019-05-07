
# coding: utf-8

# In[1]:


import pandas as pd

airline = pd.read_csv("airline.txt")
airline['CONS']=1

X = airline[['CONS','DISTANCE','DEP_DELAY']]
def cutoff(df):
    if df['ARR_DELAY'] > 15:
        return 1
    else:
        return 0
Y = airline.apply(cutoff,axis=1)


# In[2]:


import numpy as np
import scipy as sp
from scipy.optimize import fmin

def prob(b):
    return np.sum(-Y*(b[0]*X['CONS']+b[1]*X['DISTANCE']+b[2]*X['DEP_DELAY'])+np.log(1+np.exp(b[0]*X['CONS']+b[1]*X['DISTANCE']+b[2]*X['DEP_DELAY'])))

x0 = np.array([0,0,0])
betas = fmin(prob,x0,xtol=1e-8,maxfun=1000)
betas


# In[3]:


from scipy.optimize import basinhopping

betas2 = basinhopping(prob,x0,niter=10)
betas2


# In[4]:


print(np.exp(betas[1])/(1+np.exp(betas[1])))
print(np.mean(X['DISTANCE']))


# In[5]:


print(np.exp(betas[2])/(1+np.exp(betas[2])))
print(np.mean(X['DEP_DELAY']))


# In[6]:


print(np.mean(Y))


# In[7]:


from patsy import dmatrices
import statsmodels.discrete.discrete_model as sm

logit = sm.Logit(Y, X)
logit.fit().params

