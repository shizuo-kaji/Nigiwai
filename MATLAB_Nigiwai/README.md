# Nigiwai score from trajectory
Written by Shizuo KAJI
Minors modificaios are added by Mohamed 
## Licence
MIT Licence

## Requirements
- python 3: [Anaconda](https://anaconda.org) is recommended

## Usage
- basic usage
```
python nigiwai.py -i twoShops_btw.txt
```
results will be found under "output".
- advanced usage
```
python nigiwai.py -i twoShops_btw.txt --start_frame 100 --end_frame 150 --frame_skip 1.0 --alpha 0.8 -lW 0.1 -lC 5.0
```
- for brief usage
```
python nigiwai.py -h
```
- To set ROI:
* Add ROI data lines on the top of the file "twoShops_btw.txt", each line represents data for a stationary pedestrian.
* Put (-1) under the field of 'pedestrianId', 'endTime-PID1' and 'targetId-PID2'. 
* The (-1) will be adjusted by the code to the appropriate value.

- To conver images to a video
```
    ffmpeg -pattern_type glob -i 'two*.jpg' -pix_fmt yuv420p -s 300x500 out.mp4
```
