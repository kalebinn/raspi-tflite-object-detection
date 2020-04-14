# How to set up the project without an Edge TPU
This project can be used as an extention for other projects such as smart street light cameras, dash cams for cars, UAV surveilance, etc. As such it was designed to be as expandable as possible. Here we will provide a step-by-step guide on how to run our project on your Raspberry Pi 4 without the Edge TPU described in the project. If you'd like to utilize the TPU in your project please refer to the other set up file [linked here](./set_up_instructions-TPU.md)

## Step 1. Update your Raspberry Pi
You will need a Raspberry Pi 4 (64-bit ARM processor) running Raspian.  

First, update your Raspberry Pi. Especially if it is your first time booting or you have no used it in a while.  
```bash
sudo apt-get update  
sudo apt-get upgrade  
```  
## Step 2. Install virtualenv 
Using a virual environment is recommended when developing multiple python projects on the same machine. If you have not done so yet, install a python virtual environment:  
```bash
pip3 install virtualenv
```
If you do not want a virtual environment, you may skip this step and skip any future steps that ask you to activate the environment.
## Step 3. Get this repository and start a virtual environment
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
## Step 4. Setting up the dependencies
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
