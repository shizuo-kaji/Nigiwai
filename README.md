# Nigiwai score from trajectory of agents
This is a companion code for the paper 
- Measuring “Nigiwai” from Pedestrian Movement by M. A. Abdelwahab, S. Kaji, M. Hori, S. Takano, Y. Arakawa, R. Taniguchi, to appear in IEEE Access.

## Licence
MIT Licence

## Requirements
- python 3: [Anaconda](https://anaconda.org) is recommended
- numba, tqdm: install with `conda install numba tqdm`
- opencv: install with `pip install opencv-python`

## Usage
There are two versions of the main class.
- The class Nigiwai defined in nigiwai.py
takes the current coordinates of all the agents and computes the Nigiwai score.
This is suitable when the total number of agents is not big and known in advance.
- The class Nigiwai_dict defined in nigiwai_dict.py
takes the coordinates and the IDs of the agents which currently exist, and computes the Nigiwai score.
This is suitable for online computation when the total number of agents is not known in advance.

### trajectory created by the Vadare simulator
[Vadare](http://www.vadere.org/) is an open-source crowd simulator.
Some sample project files are founbd under "Vadare_projects". 
Simlulated trajectories are saved in "postvis.traj" which is a text file.

- Nigiwai can be computed for _shopping_100.txt_ (which is renamed from postvis.traj) by
```
python nigiwai_vadare.py -i data/shopping_100.txt
```
results will be found under "output".
By default, the class _Nigiwai_ is used. 
If the command-line argument (--online) is specified, the class Nigiwai_dict is used instead.

- We can specify hyper-parameters and so on by:
```
python nigiwai_vadare.py -i data/shopping_100.txt --start_frame 100 --end_frame 150 --frame_skip 1.0 --alpha 0.8 -lV 0.1 -lC 5.0
```

- for a brief description of the command-line arguments
```
python nigiwai_vadare.py -h
```

- To convert images into a video, use ffmpeg:
```
ffmpeg -pattern_type glob -framerate 10 -i 'shopping_100*.jpg' -pix_fmt yuv420p -s 300x500 out.mp4
```

### Trajectory detected by YOLOv4
[YOLOv4](https://github.com/AlexeyAB/darknet) is a deep-learning based algorithm to detect trajectories from a video.

We included the [MOT16-04 video](https://motchallenge.net/data/MOT16/) as an example.
- We can compute Nigiwai by
```
python nigiwai_Yolo.py -i data/MOT16-04.csv
```

- The csv file consists of lines with
```
frame_id,pedestrianId,X,Y,W,H,conf,Xt,Yt
```
where X,Y are the coordinates of the pedestrian in the video,
W,H are the width and the height of the bounding box,
and Xt,Yt are the coordinates of the pedestrian in the real field (which can be obtained by a projective transformation according to the camera configuration).


- To output video overlay, specify the original video file as in
```
python nigiwai_Yolo.py -i data/MOT16-04.csv -v data/MOT16-04.mp4  
```
The video file MOT16-04.mp4 can be downloaded at [MOT16](https://motchallenge.net/data/MOT16/).

- for a brief description of the command-line arguments
```
python nigiwai_Yolo.py -h
```
