
# coding: utf-8

# In[1]:


import pandas as pd

airline = pd.read_csv("airline.txt")
airline['CONS']=1
airline.head()


# In[2]:


day_dummies = pd.get_dummies(airline["DAY_OF_WEEK"],prefix='DAY')
day_dummies = day_dummies.drop('DAY_7',axis=1)
day_dummies.head()


# In[3]:


X = pd.concat([airline[['CONS','DISTANCE','DEP_DELAY']],day_dummies],axis=1)
Y = airline['ARR_DELAY']


# In[4]:


import numpy as np

b_ols = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
print(b_ols)


# In[5]:


import scipy as sp
from scipy.optimize import fmin

def f(b):
    return np.sum((Y-b[0]*X['CONS']-b[1]*X['DISTANCE']-b[2]*X['DEP_DELAY']-b[3]*X['DAY_1']-b[4]*X['DAY_2']
            + -b[5]*X['DAY_3']-b[6]*X['DAY_4']-b[7]*X['DAY_5']-b[8]*X['DAY_6'])**2)

x0 = np.array([0,0,0,0,0,0,0,0,0])
betas = fmin(f,x0,xtol=1e-8,maxfun=10000)
betas


# In[8]:


from scipy.optimize import basinhopping

betas2 = basinhopping(f,x0,niter=10)
betas2.x

