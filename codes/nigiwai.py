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
    d = np.where(np.logical_or(x[:,np.newaxis]==-1,x[np.newaxis,:]==-1), np.inf, d)   # x == -1 means the person is not in the field
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
parser.add_argument('--frame_skip', '-fs', default=1.0, type=float, help='sampling interval (unit time)')
parser.add_argument('--alpha', '-a', default=0.8, type=float, help='weight for moving average')
parser.add_argument('--lambda_W', '-lW', default=1.0, type=float, help='weight')
parser.add_argument('--lambda_C', '-lC', default=2.0, type=float, help='weight')
parser.add_argument('--scale', '-sc', default=10.0, type=float, help='spatial scaling')
parser.add_argument('--amplitude', '-amp', default=10000.0, type=float, help='multiplier for Nigiwai score')

args = parser.parse_args()

args.output = os.path.join(args.outdir,os.path.splitext(os.path.basename(args.input))[0])
print("Find outputs in {}".format(args.outdir))
os.makedirs(args.outdir, exist_ok=True)

if not args.bg_img:
    args.bg_img = os.path.splitext(args.input)[0]+".png"
    
# %%
df = pd.read_csv(args.input,header=0,delim_whitespace=True,dtype='f8')
n = int(df['pedestrianId'].max())
df['pedestrianId'] -= 1 # PID start with 0

if args.end_frame<0:
    args.end_frame = int(df['endTime-PID1'].max()/args.frame_skip)

X = np.full((n,args.end_frame+1),-1.0)  # (pid,frame),  x == -1 means the person is not in the field
Y = np.full((n,args.end_frame+1),-1.0)  # (pid,frame)

# %%  interpolate trajectory
roi_ids = []
for index, row in df.iterrows():
    pid = int(row['pedestrianId'])
    if row['targetId-PID2'] < 0:  # this is considered to be ROI and it does not affect Nigiwai score
        roi_ids.append(pid)
    start_fr = max(round(row['simTime']/args.frame_skip),args.start_frame)
    end_fr = min(round(row['endTime-PID1']/args.frame_skip)+1,args.end_frame)
    n_fr = end_fr-start_fr
    for fr in range(n_fr):
        X[pid,start_fr+fr] = args.scale*(fr*row['endX-PID1'] + (n_fr-fr)*row['startX-PID1'])/n_fr
        Y[pid,start_fr+fr] = args.scale*(fr*row['endY-PID1'] + (n_fr-fr)*row['startY-PID1'])/n_fr

print("Number of pedestrians: {}, Number of ROIs: {}".format(n-len(roi_ids),len(roi_ids)))

# %% compute Nigiwai score for each frame
D = np.full((n,n),np.inf)  # (pid,pid)
nigiwai = []
total_nigiwai = []
bg_img = ImageOps.flip(Image.open(args.bg_img))  # image upside down
for fr in range(args.start_frame,args.end_frame):
#    print("Computing frame: ",fr)
    # draw image
    img = bg_img.copy()
    draw = ImageDraw.Draw(img)
    nD = dist(X[:,fr],Y[:,fr],D,args.alpha)
    nD[:,roi_ids]=np.inf  # ROI IDs does not contribute to Nigiwai
    V=np.abs(np.where(np.logical_and(nD != np.inf, D != np.inf), nD-D, 0))
    np.nan_to_num(V, nan=0, copy=False)
    D = nD
    nigiwai_fr = np.zeros(n)
    for i in range(n):
        nigiwai_fr[i] = args.amplitude*np.sum(np.exp(-V[i,:]*args.lambda_W)/((D[i,:]+args.lambda_C)**2))
        s=6
        x,y=X[i,fr],Y[i,fr]
        if x>=0:
            col = np.clip(nigiwai_fr[i], 0, 255).astype(np.uint8)
            draw.ellipse((x-s, y-s, x+s, y+s), fill =(col,0,0), outline=(col,0,10),width=1)
#            print(x,y,col)

    total_nigiwai.append(np.sum(nigiwai_fr))
    draw.text((10,10), "Total Nigiwai {}".format(total_nigiwai[-1]), (0, 0, 255))
    nigiwai.append(nigiwai_fr)
    img.save('{}_{:0>4}.jpg'.format(args.output,fr), quality=85)


# save to csv
np.savetxt(args.output+'.csv',np.array(nigiwai),delimiter=",")
plt.plot(np.arange(args.start_frame,args.end_frame), np.array(total_nigiwai))
plt.savefig(args.output+'_total.jpg')


# %%

# %%
