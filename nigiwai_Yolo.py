#!/usr/bin/env python
import os, gzip, glob
import pandas as pd
import argparse
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import json


# load Yolo output csv
def load_Yolo(df,scale, start_frame, end_frame, frame_skip=1, min_length=0):
    # remove pedestrians who appear only a small number of frames
    for pid, count in df['pedestrianId'].value_counts().iteritems():
        if count < min_length:
            df = df[ df['pedestrianId'] != pid ]

    df['pedestrianId'] = df.groupby(['pedestrianId']).ngroup()
    df = df.sort_values(['pedestrianId', 'frame_id'], ascending=[True, True])

    # spacial scaling
    for col in ['Xt','Yt']:
        df[col] *= scale  #spatial scale

    n = int(df['pedestrianId'].max()+1)
    print("Numbner of pedestrians: ",n)

    df['frame_id'] -= 1 # frame start with 0

    X = np.full((n,end_frame),np.nan)  # (pid,frame),  x == -1 means the person is not in the field
    Y = np.full((n,end_frame),np.nan)  # (pid,frame)
    W = np.full((n,end_frame),np.nan)  # (pid,frame)
    H = np.full((n,end_frame),np.nan)  # (pid,frame)
    #CONF = np.full((n,end_frame),-1,dtype='float16')  # (pid,frame)
    Xt = np.full((n,end_frame),np.nan)  # Transformed
    Yt = np.full((n,end_frame),np.nan)  # Transformed
    # %%  read trajectory
    print("Interpolating pedestrian trajectory...")
    for pid in tqdm(range(n)):
        dfs = df[df['pedestrianId'] == pid]
        dfs.reset_index(drop=True, inplace=True)
        pid = int(pid)
        for j in range(len(dfs)-1):
            fr=int(dfs.at[j,'frame_id'])
            n_fr = int(dfs.at[j+1,'frame_id'])
            wo = float(dfs.at[j,'W'])
            ho = float(dfs.at[j,'H'])
            if fr < end_frame and wo*ho<25000: ## not too big bbox
                X[pid,fr:n_fr] = np.linspace(dfs.at[j,'X'],dfs.at[j+1,'X'],n_fr-fr)
                Y[pid,fr:n_fr] = np.linspace(dfs.at[j,'Y'],dfs.at[j+1,'Y'],n_fr-fr)
                W[pid,fr:n_fr] = np.linspace(dfs.at[j,'W'],dfs.at[j+1,'W'],n_fr-fr)
                H[pid,fr:n_fr] = np.linspace(dfs.at[j,'H'],dfs.at[j+1,'H'],n_fr-fr)
                # CONF[pid,fr] = (dfs.at[j,'conf'])
                Xt[pid,fr:n_fr] = np.linspace(dfs.at[j,'Xt'],dfs.at[j+1,'Xt'],n_fr-fr)
                Yt[pid,fr:n_fr] = np.linspace(dfs.at[j,'Yt'],dfs.at[j+1,'Yt'],n_fr-fr)
    a=range(0,Xt.shape[1],args.frame_skip) 
    return(X[:,a],Y[:,a],W[:,a],H[:,a],Xt[:,a],Yt[:,a],n)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--video', '-v', default=None, type=str, help='video file name')
parser.add_argument('--outdir', '-o', default=None, type=str, help='name of the output dir')
parser.add_argument('--input', '-i', default='data/MOT04.csv', type=str, help='csv trajectory file')
parser.add_argument('--start_frame', '-sf', default=0, type=int, help='start frame')
parser.add_argument('--end_frame', '-ef', default=-1, type=int, help='end frame')
parser.add_argument('--frame_skip', '-fs', default=4, type=int, help='sampling interval (unit number of frames)')
parser.add_argument('--frame_per_sec', '-fps', default=24, type=int, help='number of frames per second')
parser.add_argument('--alpha', '-a', default=0.9, type=float, help='weight for moving average for coordinates')
parser.add_argument('--beta', '-b', default=0.1, type=float, help='weight for moving average for velocity')
parser.add_argument('--gamma', '-g', default=0.9, type=float, help='weight for moving average for nigiwai')
parser.add_argument('--lambda_V', '-lV', default=0.1, type=float, help='weight')
parser.add_argument('--lambda_D', '-lD', default=0.8, type=float, help='weight')
parser.add_argument('--min_V', '-mV', default=0.5, type=float, help='velocity below this value will be regarded as 0')
parser.add_argument('--min_D', '-mD', default=1.0, type=float, help='distance below this value will be regarded as 0')
parser.add_argument('--scale', '-sc', default=1/80, type=float, help='spatial scaling')
parser.add_argument('--amplitude', '-amp', default=5000, type=float, help='multiplier for Nigiwai score')
parser.add_argument('--min_length', '-ml', default=20, type=int, help='Minimum trajectory length')
parser.add_argument('--write_image_every', '-w', type=int, default=4)
parser.add_argument('--online', action='store_true', help="online update of pedestrians (maybe slower)")
parser.add_argument('--monitor', '-m', default=[30,13,46], type=int, nargs="*", help='perdestrian id to monitor')
args = parser.parse_args()

if args.online:
    from nigiwai_dict import Nigiwai_dict, output_files
else:
    from nigiwai import Nigiwai

fn= os.path.splitext(os.path.basename(args.input))[0]
if args.outdir is None:
    args.outdir = "output_C{}W{}".format(args.lambda_D,args.lambda_V)
args.output = os.path.join(args.outdir,fn)
print("Find outputs in {}".format(args.outdir))
os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir,"args.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

# %%
df = pd.read_csv(args.input,header=0,dtype='f8')
if args.end_frame<0:
    args.end_frame = int(df['frame_id'].max())+1
X, Y, W, H, Xt, Yt, n = load_Yolo(df,args.scale, args.start_frame, args.end_frame, args.frame_skip, min_length=args.min_length)

# %% compute Nigiwai score for each frame
if args.video:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(args.video)
    for ff in range(args.start_frame+1):
        ret, img = cap.read()
    height, width, layers = img.shape
    video_name = args.output+'_alp{}_Wv{}_Wd{}_amp{}0o.mp4'.format(args.alpha,args.lambda_V,args.lambda_D,args.amplitude)
    fourcc =cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 15, (width,height))

nM = len(args.monitor)
velocity=[ [] for i in range((nM*(nM-1))//2) ]
distance=[ [] for i in range((nM*(nM-1))//2) ]
Nig =[ [] for i in range(nM) ]

# frame skip
if args.online:
    Ng = Nigiwai_dict(args.alpha, args.beta, args.gamma, args.lambda_D, args.lambda_V, args.min_D, args.min_V, args.amplitude)
else:
    Ng = Nigiwai(n,n,args.alpha, args.beta, args.gamma, args.lambda_D, args.lambda_V, args.min_D, args.min_V, args.amplitude)

t_range = np.arange(args.start_frame//args.frame_skip+1,int(args.end_frame/args.frame_skip))
print("Computing Nigiwai for frames {} -- {}".format(args.start_frame,args.end_frame))
for fr in tqdm(t_range):
    if args.video:
        for ff in range(args.frame_skip):
            ret, img = cap.read()
        if fr % args.write_image_every == 0:
            img1 = img.copy()
            img2 = img.copy()

    # nigiwai computation
    if args.online:
        idPed = np.where(~np.isnan(Xt[:,fr]))[0]
        XPed = np.vstack([Xt[idPed,fr],Yt[idPed,fr]]).T
        Ng.update(XPed,XPed,idPed,idPed,args.frame_skip/args.frame_per_sec)
    else:
        Ng.compute_nigiwai(Xt[:,fr],Yt[:,fr],Xt[:,fr],Yt[:,fr],args.frame_skip/args.frame_per_sec)
    if args.video and fr % args.write_image_every == 0:
        for i in range(n):
            if not np.isnan(X[i,fr]): # ignore those who are not in the field (x==-1)
                x,y,w,h=int(X[i,fr]),int(Y[i,fr]),int(W[i,fr]),int(H[i,fr])
                # conf=(CONF[i,fr])        
                col = int(np.clip(Ng.get(i), 0, 255))
                img1 = cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,col),1)
                img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,col),-1)
                img1 = cv2.putText(img1, "{:.2f}".format(Ng.get(i)), (x,y-2), font, 0.5,(255,255,255), 1, cv2.FONT_HERSHEY_PLAIN)
                img1 = cv2.putText(img1, "{:.0f}".format(i), (x,y-25), font, 0.5,(255,255,0), 1, cv2.FONT_HERSHEY_PLAIN)
                IMG=cv2.addWeighted(img1, 0.5, img2, 0.5, 1.0)
        IMG = cv2.putText(IMG,  "Frame {}".format(str(fr*args.frame_skip+1)), (10,30), font, 0.8,(0,0,255), 2, cv2.FONT_HERSHEY_PLAIN)
        video.write(IMG)
#        cv2.imwrite('{}_{:0>4}.jpg'.format(args.output,fr*args.frame_skip+1), IMG) 
#        cv2.imshow('image', IMG)
        #cv2.waitKey(30)

    # record stats
    k=0
    for i in range(len(args.monitor)):
        Nig[i].append(Ng.get(args.monitor[i]))
        for j in range(i+1,len(args.monitor)):
            distance[k].append(Ng.getD(args.monitor[i],args.monitor[j]))
            velocity[k].append(Ng.getV(args.monitor[i],args.monitor[j]))
            k += 1


cv2.destroyAllWindows() 
if args.video:
    video.release() 

## monitored pedestrians
t_range *= args.frame_skip
k=0
for i in range(len(args.monitor)):
    for j in range(i+1,len(args.monitor)):
        plt.plot(t_range,distance[k], label="{}-{}".format(args.monitor[i],args.monitor[j]))
        k += 1
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.savefig(os.path.join(args.outdir,"dist_monitor.jpg"))
plt.close()

for k in range(nM):
    plt.plot(t_range,Nig[k],label="{}".format(args.monitor[k]))
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.savefig(os.path.join(args.outdir,"nigiwai_monitor.jpg"))
plt.close()

k=0
for i in range(len(args.monitor)):
    for j in range(i+1,len(args.monitor)):
        plt.plot(t_range,velocity[k], label="{}-{}".format(args.monitor[i],args.monitor[j]))
        k += 1
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.savefig(os.path.join(args.outdir,"velocity_monitor.jpg"))
#print(velocity)

if args.online:
    output_files(Ng,os.path.join(args.outdir,"0total"))
else:
    Ng.output_files(os.path.join(args.outdir,"0total"))



















