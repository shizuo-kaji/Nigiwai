#!/usr/bin/env python

#%%
import os, gzip, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from numba import njit,jit,prange,types, typed, typeof
from numba.experimental import jitclass

def output_files(Ng, fname, n=1):
    # save scores to csv and plot a graph
    tNg = np.array(Ng.total_nigiwai)/n
    np.savetxt(fname+'_number.csv',np.array(Ng.number),delimiter=",")
    np.savetxt(fname+'_Nigiwai.csv',tNg,delimiter=",")
#    with open(fname+'.csv', 'w') as f:
#        f.write(str(Ng.nigiwai))
            
    trange=np.arange(len(Ng.total_nigiwai))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ln1=ax1.plot(trange, tNg,'C0',label='Nigiwai')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.set_xlabel('frame')
    ax1.set_ylabel('Nigiwai')
    ax1.grid(True)
    title = "avg. {}\n\n".format(sum(tNg)/(len(Ng.total_nigiwai)))
    plt.title(title)
    plt.savefig(fname+'.png')
    print(title)

@jitclass([ ('amp',types.float64), ('alpha_D', types.float64), ('alpha_V', types.float64), ('alpha_N', types.float64), 
('D_min', types.float64), ('V_min', types.float64), ('lambda_D', types.float64), ('lambda_V', types.float64), 
('nigiwai', types.DictType(types.int64,types.float64)), ('total_nigiwai', types.ListType(types.float64)),
('number', types.ListType(types.int64)),
('D', types.DictType(types.int64,types.float64)),
('V', types.DictType(types.int64,types.float64)) ])
class Nigiwai_dict(object):
    def __init__(self, alpha_D=1.0, alpha_V=1.0, alpha_N=1.0, lambda_D=1.0, lambda_V=1.0, D_min=0, V_min=0, amp=1000):
        self.amp = amp  ## Nigiwai value will be multiplied by this number
        self.alpha_D, self.alpha_V, self.alpha_N = alpha_D, alpha_V, alpha_N  ## moving average smoothing parameters
        self.lambda_D, self.lambda_V = lambda_D, lambda_V  ## weights of distance and velocity in Nigiwai computation
        self.D_min, self.V_min = D_min, V_min
        self.nigiwai = typed.Dict.empty(types.int64,types.float64)  ## latest Nigiwai scores for each agent
        self.total_nigiwai = typed.List.empty_list(types.float64) ## history of the sum of Nigiwai scores
        self.number=typed.List.empty_list(types.int64)   ## history of the number of agents
        self.D = typed.Dict.empty(types.int64,types.float64)  # (pid,pid)   distance among agents
        self.V = typed.Dict.empty(types.int64,types.float64)  # relative velocity among agents

    def get_key(self,i,j):
        return(i + (j << 32))
    
    def get_tuple(self,k):
        return(k & 0xffffffff, k >> 32)

    # update state
    def update(self,X1,X2,ID1,ID2,frame_step=1.0):
        # update distance and relative velocity
        num_agent = 0
        nV = typed.Dict.empty(types.int64,types.float64)
        nD = typed.Dict.empty(types.int64,types.float64)
        nigiwai_fr = typed.Dict.empty(types.int64,types.float64)
        for u in prange(len(ID1)):
            i,x1 = ID1[u],X1[u]
            num_agent += 1
            ng = 0
            for j,x2 in zip(ID2,X2):
                d = np.sqrt(np.sum( (x1-x2)**2 ))
                if d > 1e-20:
                    k = self.get_key(i,j)
                    if k in self.D:
                        d = (1-self.alpha_D)*self.D[k]+self.alpha_D*d
                        v = np.abs((d-self.D[k])/frame_step)
                        if k in self.V:
                            v = (1-self.alpha_V)*self.V[k] + self.alpha_V * v
                        nV[k] = v
                        ng += self.nigiwai_vel(v)*self.nigiwai_D(d)
                    nD[k] = d
            nigiwai_fr[i] = self.amp * ng
        self.number.append(num_agent)
        self.V = nV
        self.D = nD
        sum_ng = 0
        for i in self.nigiwai:
            if i in nigiwai_fr:
                nigiwai_fr[i] = self.alpha_N * nigiwai_fr[i] + (1-self.alpha_N)*self.nigiwai[i]
            else:
                nigiwai_fr[i] = (1-self.alpha_N)*self.nigiwai[i]
        for i in nigiwai_fr:
            sum_ng += nigiwai_fr[i]
        self.nigiwai = nigiwai_fr
#        print(self.nigiwai)
        self.total_nigiwai.append(sum_ng)
        return

    def nigiwai_vel(self, Vi):
        return( 1.0/(np.maximum(Vi,self.V_min)/self.lambda_V+1)**2 )
#        return( 1.0/(np.maximum(Vi-self.V_min,0)/self.lambda_V+1)**2 )

    def nigiwai_D(self, Di):
        return(np.exp(-np.maximum(Di,self.D_min)/self.lambda_D))
#        return(np.exp(-np.maximum(Di-self.D_min,0)/self.lambda_D))

    def get(self,i):
        if i in self.nigiwai:
            return self.nigiwai[i]
        return 0

    def getD(self,i,j):
        k = self.get_key(i,j)
        if k in self.D:
            return self.D[k]
        return np.nan

    def getV(self,i,j):
        k = self.get_key(i,j)
        if k in self.V:
            return self.V[k]
        return np.nan
