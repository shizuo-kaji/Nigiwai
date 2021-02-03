#!/usr/bin/env python
import os, gzip, glob
import pandas as pd
import argparse
import numpy as np
from PIL import ImageFont, Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2


# ROI placed on a uniform grid
def ROI_grid(width, height, roi_nx_ny):
    rX, rY = np.meshgrid(np.linspace(0,width,roi_nx_ny[0]),np.linspace(0,height,roi_nx_ny[1]))
    rX = rX.ravel()
    rY = rY.ravel()
    return(rX,rY)

# %%  interpolate trajectory from Vadare output txt
def load_vadere(df, scale, start_frame, end_frame, frame_skip):
    nPed = int(df['pedestrianId'].max())
    df['pedestrianId'] -= 1 # PID start with 0
    X = np.full((nPed,end_frame),np.nan)  # (pid,frame),  x == nan means the person is not in the field
    Y = np.full((nPed,end_frame),np.nan)  # (pid,frame)
    print("Interpolating pedestrian trajectory...")
    for pid,tid,st,et,sx,ex,sy,ey in tqdm(zip(df['pedestrianId'],df['targetId-PID2'],df['simTime'],df['endTime-PID1'],df['startX-PID1'],df['endX-PID1'],df['startY-PID1'],df['endY-PID1']),total=df.shape[0]):
        pid = int(pid)
        start_fr = max(int(round(st/frame_skip)),start_frame)
        end_fr = min(int(round(et/frame_skip))+1,end_frame)
        X[pid,start_fr:end_fr] = np.linspace(sx,ex, end_fr-start_fr)
        Y[pid,start_fr:end_fr] = np.linspace(sy,ey, end_fr-start_fr)
    return(scale*X,scale*Y)


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='supermarket.txt', type=str, help='Vadere trajectory file')
parser.add_argument('--outdir', '-o', default='output', type=str, help='name of the output dir')
parser.add_argument('--background', '-bg', default='', type=str, help='background image/video')
parser.add_argument('--start_frame', '-sf', default=0, type=int, help='start frame')
parser.add_argument('--end_frame', '-ef', default=-1, type=int, help='end frame')
parser.add_argument('--frame_skip', '-fs', default=1.0, type=float, help='sampling interval (unit time)')
parser.add_argument('--alpha', '-a', default=0.9, type=float, help='weight for moving average for coordinates')
parser.add_argument('--beta', '-b', default=0.9, type=float, help='weight for moving average for velocity')
parser.add_argument('--gamma', '-g', default=1, type=float, help='weight for moving average for nigiwai')
parser.add_argument('--lambda_V', '-lV', default=0.01, type=float, help='weight')
parser.add_argument('--lambda_D', '-lD', default=4.0, type=float, help='weight')
parser.add_argument('--min_V', '-mV', default=0.01, type=float, help='velocity below this value will be regarded as 0')
parser.add_argument('--min_D', '-mD', default=4.0, type=float, help='distance below this value will be regarded as 0')
parser.add_argument('--scale', '-sc', default=10.0, type=float, help='spatial scaling')
parser.add_argument('--amplitude_ped', '-ap', default=5000.0, type=float, help='multiplier for pedestrian Nigiwai score')
parser.add_argument('--amplitude_roi', '-ar', default=5000.0, type=float, help='multiplier for ROI Nigiwai score')
parser.add_argument('--write_image_every', '-w', type=int, default=1)
parser.add_argument('--output_video', '-ov', action='store_true')
parser.add_argument('--online', action='store_true', help="online update of pedestrians (maybe slower)")
parser.add_argument('--roi_nx_ny', '-rn', default=[48,80], type=int, nargs="*", help='Number of the uniform ROI points nx,ny')
args = parser.parse_args()

if args.online:
    from nigiwai_dict import Nigiwai_dict, output_files
else:
    from nigiwai import Nigiwai

fn, ext = os.path.splitext(args.input)
args.output = os.path.join(args.outdir,os.path.basename(fn))
print("Find outputs in {}".format(args.outdir))
os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir,"args.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

if not args.background:
    args.background = fn+".jpg"

if args.output_video:
    bg_img = cv2.imread(args.background)[::-1]  # flip
    height,width,_  = bg_img.shape
    fourcc =cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(args.output+'.mp4', fourcc, 15, (width,height))
else:
    bg_img = ImageOps.flip(Image.open(args.background).convert('RGB'))  # image upside down
    width,height  = bg_img.size
print("Area size: ",width,height)

# %%
fn, ext = os.path.splitext(args.input)

df = pd.read_csv(args.input,header=0,delim_whitespace=True,dtype='f8')
if args.end_frame<0:
    args.end_frame = int(round(df['endTime-PID1'].max()/args.frame_skip))

rX, rY = ROI_grid(width, height, args.roi_nx_ny)
X, Y = load_vadere(df, args.scale, args.start_frame, args.end_frame, args.frame_skip)
nPed, nROI = len(X), len(rX)
if args.online:
    NgPed = Nigiwai_dict(args.alpha, args.beta, args.gamma, args.lambda_D, args.lambda_V, args.min_D, args.min_V, args.amplitude_ped)
    NgROI = Nigiwai_dict(args.alpha, args.beta, args.gamma, args.lambda_D, args.lambda_V, args.min_D, args.min_V, args.amplitude_roi)
else:
    NgPed = Nigiwai(nPed,nPed,args.alpha, args.beta, args.gamma, args.lambda_D, args.lambda_V, args.min_D, args.min_V, args.amplitude_ped)
    NgROI = Nigiwai(nROI,nPed,args.alpha, args.beta, args.gamma, args.lambda_D, args.lambda_V, args.min_D, args.min_V, args.amplitude_roi)

print("Number of pedestrians: {}, Number of ROIs: {}".format(nPed,nROI))

# %% compute Nigiwai score for each frame

s=2
print("Computing Nigiwai for frames {} -- {}".format(args.start_frame,args.end_frame))
for fr in tqdm(range(args.start_frame+1,args.end_frame-1)):
    if args.online:
        idPed = np.where(~np.isnan(X[:,fr]))[0]
        idRoi = np.arange(len(rX))
        XPed = np.vstack([X[idPed,fr],Y[idPed,fr]]).T
        XRoi = np.vstack([rX,rY]).T
        NgPed.update(XPed,XPed,idPed,idPed,args.frame_skip)
        NgROI.update(XRoi,XPed,idRoi,idRoi,args.frame_skip)
    else:
        NgPed.compute_nigiwai(X[:,fr],Y[:,fr],X[:,fr],Y[:,fr],args.frame_skip)
        NgROI.compute_nigiwai(rX,rY,X[:,fr],Y[:,fr],args.frame_skip)

    # draw image
    if fr % args.write_image_every == args.write_image_every-1:
        if args.output_video:
            img1 = bg_img.copy()
            img2 = bg_img.copy()
        else:
            img = Image.new('RGBA', bg_img.size, (255,255,255,0))
            draw = ImageDraw.Draw(img)

        for i in range(nROI):
            x,y=int(rX[i]),int(rY[i])
            col = int(np.clip(NgROI.get(i), 0, 255))
            if args.output_video:
                img2 = cv2.rectangle(img2,(x-s,y-s),(x+s,y+s),(0,0,col),-1)
            else:
                draw.rectangle((x-s, y-s, x+s, y+s), fill =(col,0,0,col), outline=(col,0,0,col),width=1)

        for i in range(nPed):
            if not np.isnan(X[i,fr]):
                x,y=int(X[i,fr]),int(Y[i,fr])
                col = int(np.clip(NgPed.get(i), 0, 255))
                if args.output_video:
                    img1 = cv2.circle(img1, (x,y), s, (0,0,col), thickness=-1)
                    img2 = cv2.circle(img2, (x,y), s, (0,0,col), thickness=-1)
                else:
                    draw.ellipse((x-s, y-s, x+s, y+s), fill =(col,0,255-col,255), outline=(col,0,255-col,255),width=1)

        if args.output_video:
            img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
            img = cv2.putText(img,  "Frame {}".format(str(fr*args.frame_skip)), (200,10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0), 1, cv2.FONT_HERSHEY_PLAIN)
            img = cv2.putText(img,  "ROI_Nigiwai {:.2f}".format(NgROI.total_nigiwai[-1]/nROI), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0), 1, cv2.FONT_HERSHEY_PLAIN)
            img = cv2.putText(img,  "Ped_Nigiwai {:.2f}".format(np.sqrt(NgPed.total_nigiwai[-1])), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0), 1, cv2.FONT_HERSHEY_PLAIN)
            cv2.imwrite('{}_{:0>4}.jpg'.format(args.output,fr), img) 
            video.write(img)
        else:
        # PIL ver. 
            draw.text((240,10), "Frame {}, #Ped {}".format(str(fr), NgPed.number[-1]), (0, 0, 255,255))
            draw.text((10,10), "ROI_Nigiwai {:.2f}".format(NgROI.total_nigiwai[-1]/nROI), (0, 0, 255,255))
            draw.text((10,25), "Ped_Nigiwai {:.2f}".format(np.sqrt(NgPed.total_nigiwai[-1])), (0, 0, 255,255))
            #composite = Image.alpha_composite(bg_img, img).convert('RGB')
            outimg = bg_img.copy()
            outimg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            outimg.save('{}_{:0>4}.jpg'.format(args.output,fr), quality=80)

if args.output_video:
    video.release() 

if args.online:
    output_files(NgPed,args.output+"_Ped")
    output_files(NgROI,args.output+"_ROI",nROI)
else:
    NgPed.output_files(args.output+"_Ped",trange=np.arange(args.start_frame+1,args.end_frame-1))
    NgROI.output_files(args.output+"_ROI",trange=np.arange(args.start_frame+1,args.end_frame-1), n=nROI)


