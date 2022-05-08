# SUO-SLAM

This repository hosts the code for our CVPR 2022 paper
"Symmetry and Uncertainty-Aware Object SLAM for 6DoF Object Pose Estimation".
[ArXiv link](http://arxiv.org/abs/2205.01823).

<p align="center">
  <img width="640" height="160" src="./assets/suo_slam_demo.gif">
</p>

## Citation
If you use any part of this repository in an academic work, please cite our paper as:

```
@inproceedings{Merrill2022CVPR,
  Title      = {Symmetry and Uncertainty-Aware Object SLAM for 6DoF Object Pose Estimation},
  Author     = {Nathaniel Merrill and Yuliang Guo and Xingxing Zuo and Xinyu Huang and Stefan Leutenegger and Xi Peng and Liu Ren and Guoquan Huang},
  Booktitle  = {2022 Conference on Computer Vision and Pattern Recognition (CVPR)},
  Year       = {2022},
  Address    = {New Orleans, USA},
  Month      = jun,
}
```

## Installation
<details>
<summary>Click for details...</summary>
This codebase was tested on Ubuntu 18.04.
To use the BOP rendering (i.e. for keypoint labeling) install
```
sudo apt install libfreetype6-dev libglfw3
```

You will also need a python environment that contains the required packages. To 
see what packages we used, check out the list of requirements in `requirements.txt`.
They can be installed via `pip install -r requirements.txt`
</details>

## Preparing Data
<details>
<summary>Click for details...</summary>
### Datasets
To be able to run the training and testing (i.e. single view or with SLAM),
 first decide on a place to download the data to.
The disk will need a few hundred GB of space for all the data (at least 150GB for download and more
to extract).
All of our code expects the data to be in a local directory `./data`, but you 
can of course symlink this to another location (perhaps with more disk space).
So, first of all, in the root of this repo run 
```
$ mkdir data
```
or to symlink to an external location
```
$ ln -s /path/to/drive/with/space/ ./data
```

You can pick and choose what data you want to download (for example just T-LESS or YCBV).
Note that all YCBV and TLESS downloads have our keypoint labels packaged along with the data.
Download the following google drive links into `./data` and extract them.
- [YCBV full dataset](https://drive.google.com/file/d/1C-CkqYiCC-PqySL70QX7k_YQzGccX6dG/view?usp=sharing)
- [YCBV eval-only dataset](https://drive.google.com/file/d/17aLUdsfNZ98xinCf1YJzciVE4xwUcgnz/view?usp=sharing)
- [T-LESS dataset](https://drive.google.com/file/d/1h15UYWiLYmwTJi-hh0t5EFA77H1ElSLa/view?usp=sharing)
- [Saved detections (eval only)](https://drive.google.com/file/d/1WjEUgQDs34U63vlPXkd7SOVco58Jvn6F/view?usp=sharing)
- [VOC dataset (training only)](https://drive.google.com/file/d/1QNCRac2MxbJALEHmpKW1D6Pz0oCeDTAy/view?usp=sharing)

When all is said and done, the tree should look like this
```
$ cd ./data && tree --filelimit 3
.
├── bop_datasets
│   ├── tless 
│   └── ycbv 
├── saved_detections
└── VOCdevkit
    └── VOC2012
```

### Pre-trained models
You can download the pretrained models anywhere, but I like to keep them
in the `results` directory that is written to during training.
- [Pre-trained models](https://drive.google.com/file/d/1mZMqKjE2cqpLQWHKaDo9_sjMmYBXDcu5/view?usp=sharing)
</details>

## Training
<details>
<summary>Click for details...</summary>

First set the default arguments in `./pkpnet/args.py` for your username if desired, then execute
```
$ ./train.py
```
with the appropriate arguments for your filesystem. You can also run
```
$ ./train.py -h
```
for a full list of arguments and their meaning.
Some important args are `batch_size`, which is the number of images loaded for each 
training batch. Note that there may be a variable number of objects in each image,
and the objects are all stacked together into one big batch to run the network --
so the actual batch size being run might be multiple times `batch_size`. In
order to keep `batch_size` reasonably large, we provide another arg called `truncate_obj`,
which, as the help says, truncates the object batches to this number if it exceeds it.
We recommend that you start with a large batch size so that you can find out the maximum `truncate_obj`
for you GPUs, then reduce the batch size until there are little to no warnings about too many objects
being truncated.
</details>

## Evaluation

<details>
<summary>Click for details...</summary>

Before you can evaluate in a single-view or SLAM fashion, you will need to build
the thirdparty libraries for PnP and graph optimization.
First make sure that you have [CERES solver](http://ceres-solver.org/installation.html) installed.
The run 
```
$ ./build_thirdparty.sh
``` 

### Reproducing Results
To reproduce the results of the paper with the pretrained models, check out the scripts under
the `scripts` directory:
```
eval_all_tless.sh  eval_all_ycbv.sh  make_video.sh
```
These will reproduce most of the results in the paper as well as any video clips you want.
You may have to change the first few lines of each script.
Note that these examples can also show you the proper arguments if you want to 
run from command line alone.

Note that for the T-LESS dataset, we use the thirdparty 
[BOP toolkit](https://github.com/thodan/bop_toolkit) to get the VSD error recall, which 
will show up in the final terminal output as "Mean object recall" among other numbers.

<\details>

## Labeling
<details>
<summary>Click for details...</summary>

### Overview
We manually label keypoints on the CAD model to enable some keypoints with semantic meaning.
For the full list of keypoint meanings, see the specific [README](./lib/labeling/README.md)

We provide our landmark labeling tool. Check out the script `manual_keypoints.py`.
This same script can be used to make a visualization of the keypoints as shown 
below with the `--viz` option.

![](./assets/ycbv_kp_viz.png)
![](./assets/tless_kp_viz.png)

The script will show a panel of the same object but oriented slightly differently.
The idea is that you pick the same keypoint multiple times to ensure correctness and
to get a better label by averaging multiple samples.

![](./assets/labeling_view.png)

The script will also print the following directions to follow in the terminal.

```
============= Welcome ===============
Select the keypoints with a left click!
Use the "wasd" to turn the objects.
Press "i" to zoom in and "o" to zoom out.
Make sure that the keypoint colors match between all views.
Messed up? Just press 'u' to undo.
Press "Enter" to finish and save the keypoints
Press "Esc" to just quit
```

Once you have pressed "enter", you will get to an inspection pane. 

![](./assets/inspection_panel.png)

Where the unscaled mean keypoints are on the left, and the ones scaled by covariance
is on the left, where the ellipses are the Gaussian 3-sigma projected onto the image.
If the covariance is too large, or the mean is out of place, then you may have messed up.
Again, the program will print out these directions to terminal:

```
Inspect the results!
Use the "wasd" to turn the object.
Press "i" to zoom in and "o" to zoom out.
Press "Esc" to go back, "Enter" to accept (saving keypoints and viewpoint for vizualization).
Please pick a point on the object!
```

So if you are done, and the result looks good, then press "Enter", if not then "Esc" to go back.
Make sure also that when you are done, you rotate and scale the object into the best "view pose"
(with the front facing the camera, and top facing up),
as this pose is used by both the above vizualization and the actual training code for determining
the best symmetry to pick for an initial detection.

### Labeling Tips

Even though there are 8 panels, you don't need to fill out all 8. Each keypoint just
needs at least 3 samples to sample the covariance.

We recommend that you label the same keypoint (say keypoint `i`) on all the object 
renderings first, then go to the inspection panel at the end 
of this each time so that you can easily undo a mistake for keypoint `i` with the "u" key
and not lose any work. Otherwise, if you label each object rendering completely, then
you may have to undo a lot of labelings that were not mistakes.

Also, if there is an object that you want to label a void in the CAD model, like the 
top center of the bowl, then you can use the multiple samples to your advantage, and choose
samples that will average to the desired result, since the labels are required to land on the 
actual CAD model in the labeling tool.

<\details>
