# Nigiwai score from trajectory
Written by Shizuo KAJI

## Licence
MIT Licence

## Requirements
- python 3: [Anaconda](https://anaconda.org) is recommended

## Usage
- basic usage
```
python nigiwai.py -i twoShops_btw_ROI.txt
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
- To set ROI, look at "twoShops_btw_ROI.txt". The first three perdestrians are stationary and they represent ROI.


- To conver images to a video
```
    ffmpeg -pattern_type glob -i 'two*.jpg' -pix_fmt yuv420p -s 300x500 out.mp4
```
