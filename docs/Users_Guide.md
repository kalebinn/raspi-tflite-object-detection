# Raspberry Pi Object Detection with Raspberry Pi 4 User's Guide
With support for optional Edge TPU.  
  
Make sure that you've followed the set up guide here: [Set up on your own Raspberry Pi 4](set_up_instructions.md).  

Now that you've all set up. You can being to use the detector. 

## Table of Contents
* [Step 1. Activate the virtual environment](#activate-env)
* [Step 2. Running the object detector](#running)
    * [Option A. Using the PiCamera](#with-picam)
    * [Option B. Using a video file](#with-vid)

## [Step 1. Activate the virtual environment](#activate-env)
First you will need to activate the Python 3 virtual environment if you're using one. If you are not, you may skip this step. You can activate the virtual env by typing:
```bash
cd raspi-tflite-object-detection 
source /obj-detector-env/bin/activate
```
I've named my virtual environment `obj-detector-env` but feel free to choose your own name. Just make sure to replace it in the command above. 

## [Step 2. Running the object detector](running)
At the time of writing this, the object detector only uses a quantized SSD MobileNetv1 trained on the COCO data set. I'm currently working on adding SSD MobileNetv2 and YOLOv3, both trained on the COCO data set. In the future, I'll write a short guide on how you can train your own custom tensorflow models and convert them to to a tflite model.   
  
Options for `python3 obj-detection.py`:    
| Argument | Description | Default value| 
|:---------:|:-----------|:-------------|
|--model| Specify the model you would like to use. THe name of this model must match the directory in which it is stored. | MobileNetv1_quant |
|--graph| Specify the graph file. It must be located in the model directory. If left blank, it will use the same as the model directory, but with a .tflite extention | `None` |
|--label| Specify the labels file. The script assumes it is a text file with one column. | `labels.txt`|
|--video| Video file to run inference on | `None` |
|--confidence_threshold | Minimum confidence threshold of interference for drawing bounding box | `0.5`|
|--resolution | Target resolution of the PiCamera (Can not be used in video mode) | `1280x720`|
|--use_TPU| Boolean, True if EDGE TPU is used, False otherwise | `False`| 

### [Option A. Using the PiCamera](#with-picam)
In the demostrations, I'm using a PiCamera v2, but this code also works on any PiCamera, albiet the maximum resolutions and FPS will be different. In order to run the object detector with the camera without the Edge TPU, all you have to do is:
```bash
python3 obj-detection.py 
```
  
If you're running the object detector with TPU support, you can type:
```bash
python3 obj-detection.py --use_TPU=True
```

### [Option B. Using a video file](#with-vid)
If you'd like to use a prerecorded video rather than a camera, you can just specify the video path as such:
```bash
python3 obj-detection.py --video=test.mp4
``` 

If you're using a TPU, you only need an extra argument when launching the script:
```bash
python3 obj-detection.py --video=test.mp4 --use_TPU=True
```   
