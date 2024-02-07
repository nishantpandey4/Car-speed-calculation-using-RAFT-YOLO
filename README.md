# Dependencies
test2.yml file of the conda environment is provided.  
or 
create an environment with python 3.8.

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install matplotlib tensorboard scipy opencv
# Yolo Weights
The weights for YOLO need to be downloaded and placed in the root folder.
## Demos
The code has been tested with 
For running the demo using our trained model. 
-python demo.py --model=models/enpm673_raft-kitti.pth --path=frames
For running with videos.
-python inference.py --model=models/enpm673_raft-kitti.pth
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/enpm673_raft-kitti.pth --dataset=kitti --mixed_precision
```
## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using a tensorboard
```Shell
./train_standard.sh
```
If you have an RTX GPU, training can be accelerated using mixed precision. You can expect similar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficient Implementation
You can optionally use our alternate (efficient) implementation by compiling the provided Cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, that this implementation is somewhat slower than all pairs, but uses significantly less GPU memory during the forward pass.

Other files and folders:
- weights and cfg files for yolo were obtained from https://github.com/pjreddie/darknet 
- Yolo weights can be downloaded from https://drive.google.com/drive/folders/1h9PqeZ3l5RUURJIxeNMXGt-Ilil3ngrO?usp=sharing, I got this from here https://pjreddie.com/media/files/yolov3.weights
- 1684366668.0944755.mp4 video for testing
- stream.py- for streaming the video from the pi cam from https://singleboardblog.com/real-time-video-streaming-with-raspberry-pi/ 

Note: 
- To visualize the motion vectors of the car uncomment line 51-56 in demo.py and 62-68 for inference.py
- To visualize the bounding rectangle of the car uncomment line 133-137 in demo.py and 132-135 for inference.py
- for live video streaming hardware is needed but we used line 210 of inference.py to do so. The IP address will change. 
* We downloaded images online and annotated them for YOLO training and testing but we could not get the right output so we switched to existing trained models.

Results:

YOLO Result 

![WhatsApp Image 2023-05-18 at 21 46 47](https://github.com/nishantpandey4/Car-speed-calculation-using-RAFT-YOLO/assets/127569735/68c396cb-5d5a-4ad9-a9e3-cc173be12169)

RAFT Result

![WhatsApp Image 2023-05-18 at 09 18 20](https://github.com/nishantpandey4/Car-speed-calculation-using-RAFT-YOLO/assets/127569735/e242e34f-b973-4069-a793-6da5a22cfa44)

Combined Result

![Screenshot (33)](https://github.com/nishantpandey4/Car-speed-calculation-using-RAFT-YOLO/assets/127569735/e2dc3549-a3fd-4e0f-87d5-dab31d3e1368)

### The package is based on https://github.com/princeton-vl/RAFT. 
