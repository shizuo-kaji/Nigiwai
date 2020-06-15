#!/usr/bin/env python
## reference: http://akumano.xyz/posts/arxiv-keyword-extraction-part1/
## Requirements
# conda install gensim nltk
# pip install kmapper

#%%
import os, gzip, glob
import pandas as pd
import argparse
import numpy as np
from PIL import ImageFont, Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
 

def dist(x,y,D,alpha):
    d = np.sqrt((x[:,np.newaxis]-x[np.newaxis,:])**2 + (y[:,np.newaxis]-y[np.newaxis,:])**2)
    d = np.where(np.logical_or(x[:,np.newaxis]==-1,x[np.newaxis,:]==-1), np.inf, d)
    np.fill_diagonal(d,np.inf)
    nD = np.where(D != np.inf, (1-alpha)*D+alpha*d, d)
    return(nD)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='OnlyPassing_double.txt', type=str, help='Vadere trajectory file')
parser.add_argument('--outdir', '-o', default='output', type=str, help='name of the output dir')
parser.add_argument('--bg_img', '-bg', default='', type=str, help='background image')
parser.add_argument('--start_frame', '-sf', default=0, type=int, help='start frame')
parser.add_argument('--end_frame', '-ef', default=-1, type=int, help='end frame')
parser.add_argument('--frame_skip', '-fs', default=0.4, type=float, help='sampling interval (unit time)')
parser.add_argument('--alpha', '-a', default=0.8, type=float, help='weight for moving average')
parser.add_argument('--lambda_W', '-lW', default=1.0, type=float, help='weight')
parser.add_argument('--lambda_C', '-lC', default=2.0, type=float, help='weight')
parser.add_argument('--scale', '-sc', default=10.0, type=float, help='spatial scaling')
parser.add_argument('--amplitude', '-amp', default=1.0, type=float, help='multiplier for Nigiwai score')

args = parser.parse_args()

args.output = os.path.join(args.outdir,os.path.splitext(os.path.basename(args.input))[0])
print("Find outputs in {}".format(args.outdir))
os.makedirs(args.outdir, exist_ok=True)

if not args.bg_img:
    args.bg_img = os.path.splitext(args.input)[0]+".png"
    
# %%
df = pd.read_csv(args.input,header=0,delim_whitespace=True,dtype='f8')
nPed = int(df['pedestrianId'].max())  
df['pedestrianId'] -= 1 # PID start with 0

nROI=(df['pedestrianId'][df['pedestrianId']<0]).size  # number of ROI pedestrians
df['endTime-PID1'][df['pedestrianId']<0]=df['endTime-PID1'].max()   # adjust end time for ROI to max
df['pedestrianId'][df['pedestrianId']<0]=nPed+np.arange(nROI)  # change ID for ROI from -1 to next ID of max(ID)



print("Number of pedestrians: {}, Number of ROIs: {}".format(nPed,nROI))
n=nPed+nROI # total number of pedestrians+ROI

if args.end_frame<0:
    args.end_frame = int(round(df['endTime-PID1'].max()/args.frame_skip))

X = np.full((n,args.end_frame),-1.0)  # (pid,frame)
Y = np.full((n,args.end_frame),-1.0)  # (pid,frame)

# %%  interpolate trajectory
roi_ids = []
for index, row in df.iterrows():
    pid = int(row['pedestrianId'])


    if row['targetId-PID2'] < 0:  # this is considered to be ROI and it does not affect Nigiwai score
        roi_ids.append(pid)

#    start_fr = max(int(row['simTime']/args.frame_skip),args.start_frame)
#    end_fr = min(int(row['endTime-PID1']/args.frame_skip)+1,args.end_frame)
    start_fr = max(int(round(row['simTime']/args.frame_skip)),args.start_frame) ##<<<<<= adding int
    end_fr = min(int(round(row['endTime-PID1']/args.frame_skip)),args.end_frame) ##<<<<<============ remove +1
#    if end_fr==start_fr:
#        end_fr=start_fr+1
    n_fr = end_fr-start_fr

    for fr in range(n_fr):
        X[pid,start_fr+fr] = args.scale*(fr*row['endX-PID1'] + (n_fr-fr)*row['startX-PID1'])/n_fr
        Y[pid,start_fr+fr] = args.scale*(fr*row['endY-PID1'] + (n_fr-fr)*row['startY-PID1'])/n_fr


# %% compute Nigiwai score for each frame
D = np.full((n,n),np.inf)  # (pid,pid)
nigiwai = []
total_nigiwai = []
Ped_nigiwai = []
ROI_nigiwai = []
PedNumber=[]

bg_img = ImageOps.flip(Image.open(args.bg_img))  # image upside down
for fr in range(args.start_frame,args.end_frame):
    print("Computing frame: ",fr)
    # draw image
    img = bg_img.copy()
    draw = ImageDraw.Draw(img)
    nD = dist(X[:,fr],Y[:,fr],D,args.alpha)
    nD[:,roi_ids]=np.inf  # ROI IDs does not contribute to Nigiwai
#    V=np.abs(np.where(np.logical_and(nD != np.inf, D != np.inf), nD-D, 0))
    V=np.where(np.logical_and(nD != np.inf,D != np.inf), nD-D, 0)

    np.nan_to_num(V, copy=True, nan=0)
    D = nD
    nigiwai_fr = np.zeros(n)
    for i in range(n):
        nigiwai_fr[i] = args.amplitude*np.sum(np.exp(-V[i,:]*args.lambda_W)/((D[i,:]+args.lambda_C)**2))
        s=3
        s2=6
        x,y=X[i,fr],Y[i,fr]
        if x>=0:
            if i<nPed:
                col = np.clip(nigiwai_fr[i], 0, 255).astype(np.uint8)
                draw.ellipse((x-s, y-s, x+s, y+s), fill =(col,0,0), outline=(col,0,10),width=1)
            else:
                draw.ellipse((x-s2, y-s2, x+s2, y+s2), fill =(col,0,0), outline=(0,255,0),width=1)
            

    
    PedNumber.append(np.sum(X[:,fr]>-1)-nROI)
    total_nigiwai.append(np.sum(nigiwai_fr))
    Ped_nigiwai.append(np.sum(nigiwai_fr[0:nPed]))
    ROI_nigiwai.append(np.sum(nigiwai_fr[nPed:nPed+nROI]))
    

    draw.text((240,10), "Frame {}".format(str(fr)), (0, 0, 255))
    draw.text((10,10), "Nigiwai {:.2f}".format(total_nigiwai[-1]), (0, 0, 255))
    draw.text((10,25), "Ped_Num {}".format(PedNumber[-1]), (0, 0, 255))
    nigiwai.append(nigiwai_fr)
    img.save('{}_{:0>4}.jpg'.format(args.output,fr), quality=85)


# save to csv
#np.savetxt(args.output+'.csv',np.array(nigiwai),delimiter=",")
#np.savetxt(args.output+'_Tot.csv',np.array(total_nigiwai),delimiter=",")
np.savetxt(args.output+'_Den.csv',np.array(PedNumber),delimiter=",")
np.savetxt(args.output+'_Ped.csv',np.array(Ped_nigiwai),delimiter=",")
np.savetxt(args.output+'_ROI.csv',np.array(ROI_nigiwai),delimiter=",")


plt.plot(np.arange(args.start_frame,args.end_frame), np.array(Ped_nigiwai))

plt.plot(np.arange(args.start_frame,args.end_frame), np.array(ROI_nigiwai))
plt.savefig(args.output+'.jpg')


# %%

# %%