# Object Detection using a Raspberry Pi 4 and TensorFlow Lite
With optional Google Coral.ai USB AI Accelerator (Edge TPU co-processor)  

## Description  
This is a project for EE 45900 - Microprocessors - Spring 2020 at The City College of New York. In this project we will use a Raspberry Pi to create a Real-Time Object Detector.  
Group members: Kelvin Ma and YoungHwa Min  
  
We will be using a [Raspberry Pi 4 (4 GB) model](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/?variant=raspberry-pi-4-model-b-4gb) and [Coral's USB Accelerator](https://coral.ai/products/accelerator) along with [TensorFlow-Lite](https://www.tensorflow.org/lite/) and [OpenCV](https://opencv.org/)  This project will be open source and open to modification. In fact, it was made possible by many examples and open source code. We have provided credit in the code depending on the resources. Our own original code will also be open source to all, we only ask that you provide credit.   

This code uses transfer learning model provided by TF-Lite. It uses the COCO SSD Quantized MobileNET. At the time of writing, TF-Lite only supports single shot detector models (SSD). Hence, we can not run any pretrained RCNN models.  
   
## How to run the project 
Running the project was made simple. The Edge TPU (USB Accelerator) is optional when running any code, however, there are tremendous performance gains when using the accelerator so it is always suggested. Here we will provide a step-by-step guide on how to run our project on your Raspberry Pi 4.  

### Step 1. Update your Raspberry Pi
You will need a Raspberry Pi 4 (32-bit ARM processor) running Raspian. This is neccessary for our particular project because we are using the Linux (ARM 32) installation of TensorFlow-Lite.   
First, update your Raspberry Pi. Especially if it is your first time booting or you have no used it in a while.  
```bash
sudo apt-get update  
sudo apt-get upgrade  
```  
### Step 2. Install virtualenv (if you have no previously done so)
Using a virual environment is **always** recommended when developing multiple python projects. If you have not done so yet, install a python virtual environment:  
```bash
pip3 install virtualenv
```
### Step 3. Get this repository and start a virtual environment
After you've updated your Raspberry Pi, clone this repository and navigate into the directory  
```bash
git clone https://github.com/kalebinn/raspi-tflite-object-detection.git  
cd raspi-tflite-object-detection  
```
  
Next, we will start a virtual environment before installing the dependencies (listed above).  
```bash
python -m venv obj-detector-env
```  
If you close this terminal, this virtual environment will have to reactivated as such:
```bash
cd raspi-tflite-object-detection 
source /obj-detector-env/bin/activate
```
### Step 4. Setting up the dependencies
A small python script is prewritten to set up all the dependencies. You can simple run  
```bash
python3 setup.py
```
This set up file sets up the following for OpenCV: 
* `libtiff5-dev`,
* `libjasper-dev`,
* `libpng12-dev`,
* `libavcodec-dev`,
* `libavformat-dev`,
* `libswscale-dev`,
* `libv4l-dev`,
* `libxvidcore-dev`,
* `libx264-dev`,
* `qt4-dev-tools`,
* `libatlas-base-dev`,

And uses [https://www.tensorflow.org/lite/guide/python](https://www.tensorflow.org/lite/guide/python) to install the appropriate TensorFlow-Lite iterpretor for your python version.   
**Note**: you must have Python 3.5.x, 3.6.x, or 3.7.x installed on your Raspberry Pi. Python 3.8.x is currently unsupported at the time of writing.  
