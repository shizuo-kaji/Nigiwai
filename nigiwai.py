#!/usr/bin/env python

#%%
import os, gzip, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from numba import njit,jit,prange

class Nigiwai(object):
    def __init__(self, n, m, alpha_D=1.0, alpha_V=1.0, alpha_N=1.0, lambda_D=1.0, lambda_V=1.0, D_min=0, V_min=0, amp=1000):
        self.amp = amp  ## Nigiwai value will be multiplied by this number
        self.alpha_D, self.alpha_V, self.alpha_N = alpha_D, alpha_V, alpha_N  ## moving average smoothing parameters
        self.lambda_D, self.lambda_V = lambda_D, lambda_V  ## weights of distance and velocity in Nigiwai computation
        self.D_min, self.V_min = D_min, V_min
        self.nigiwai = []   ## history of Nigiwai scores
        self.number=[]   ## history of the number of agents
        self.total_nigiwai = [] ## history of sum of Nigiwai
        self.n, self.m = n, m  # total number of agents to compute Nigiwai, and that of agenets which contribute to Nigiwai

        self.D = np.full((n,m),np.inf)  # (pid,pid)   distance among agents
        self.V = np.full((n,m),np.inf)  # relative velocity among agents

    # distance between x and Y with moving average smoothing with alpha
    def dist(self,x1,y1,x2,y2,prevD,alpha):
        d = np.sqrt((x1[:,np.newaxis]-x2[np.newaxis,:])**2 + (y1[:,np.newaxis]-y2[np.newaxis,:])**2)
        d = np.where(np.logical_or(np.isnan(x1[:,np.newaxis]),np.isnan(x2[np.newaxis,:])), np.inf, d)   # x == np.nan means the person is not in the field
        np.fill_diagonal(d,np.inf)
    #    np.seterr(all='raise')
        if alpha==1:
            nD = d
        else:
            nD = np.where(np.isinf(prevD), d, (1-alpha)*prevD+alpha*d)
        return(nD)

    def compute_nigiwai(self,X1,Y1,X2,Y2,frame_step=1.0):
        nD = self.dist(X1,Y1,X2,Y2,self.D,self.alpha_D) # distance mat in the current frame
        with np.errstate(invalid='ignore'): # suppress warning for np.inf - np.inf in nD-D
            nV= np.abs(np.where(np.logical_or(np.isinf(nD), np.isinf(self.D)), np.inf, (nD-self.D)/frame_step)) # absolute relative velocity
            self.V = np.where(np.isinf(self.V), nV, self.alpha_V * nV + (1-self.alpha_V)*self.V) 
        self.D = nD
        nigiwai_fr = np.zeros(self.n)
        num_agent = 0
        for i in range(self.n):
            if not np.isnan(X1[i]):
                num_agent += 1
                nigiwai_fr[i] = self.amp * np.sum(self.nigiwai_vel(self.V[i,:])*self.nigiwai_D(self.D[i,:]))
                if len(self.nigiwai)>0:
                    nigiwai_fr[i] = self.alpha_N * nigiwai_fr[i] + (1-self.alpha_N)*self.nigiwai[-1][i]
        self.number.append(num_agent)
        self.nigiwai.append(nigiwai_fr)
        self.total_nigiwai.append(np.sum(nigiwai_fr))

    def nigiwai_vel(self, Vi):
        return( 1.0/(np.maximum(Vi,self.V_min)/self.lambda_V+1)**2 )
#        return( 1.0/(np.maximum(Vi-self.V_min,0)/self.lambda_V+1)**2 )

    def nigiwai_D(self, Di):
        return(np.exp(-np.maximum(Di,self.D_min)/self.lambda_D))
#        return(np.exp(-np.maximum(Di-self.D_min,0)/self.lambda_D))

    def get(self,i):
        if len(self.nigiwai)>0:
            return self.nigiwai[-1][i]

    def getD(self,i,j):
        k = self.D[i,j]
        if k < np.inf:
            return k
        return np.nan

    def getV(self,i,j):
        k = self.V[i,j]
        if k < np.inf:
            return k
        return np.nan

    def output_files(self, fname, trange=None, n=1):
        tNg = np.array(self.total_nigiwai)/n
        # save scores to csv and plot a graph
        np.savetxt(fname+'_number.csv',np.array(self.number),delimiter=",")
        np.savetxt(fname+'.csv',tNg,delimiter=",")

        if trange is None:
            trange=np.arange(len(self.total_nigiwai))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1=ax1.plot(trange, tNg,'C0',label='Nigiwai')
#        ax2 = ax1.twinx()
#        ln2=ax2.plot(trange,np.average(_ROI_nigiwai,axis=1),'C1',label='ROI')
        h1, l1 = ax1.get_legend_handles_labels()
#        h2, l2 = ax2.get_legend_handles_labels()
#        ax1.legend(h1+h2, l1+l2, loc='lower right')
        ax1.set_xlabel('frame')
        ax1.set_ylabel('Nigiwai')
        ax1.grid(True)
#        ax2.set_ylabel('ROI')
        title = "avg. {}\n\n".format(np.average(tNg))
        plt.title(title)
        plt.savefig(fname+'.png')
        print(title)

