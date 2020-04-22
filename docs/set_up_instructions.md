# How to set up the project on your own Raspberry Pi
This project can be used as an extention for other projects such as smart street light cameras, dash cams for cars, UAV surveilance, etc. As such it was designed to be as expandable as possible. Here we will provide a step-by-step guide on how to set up this project on your Raspberry Pi 4. 

## Table of Contents
* [Step 1. Update your Raspberry Pi](#step_1)
* [Step 2. Install Virualenv](#step_2)
* [Step 3. Get this repository and start a virtual environment](#step_3)
* [Step 4. Setting up the dependencies](#step_4)
    * [Option A. Without TPU](#step_4a)
    * [Option B. With TPU](#step_4b)
    * [Option C. With TPU operating at 2-times default frequency](#step_4c)

After you're done setting up, a short User's Guide was written for quick later reference. Once you're finished with this document, you can go ahead and read the [User's Guide](Users_Guide.md).


## [Step 1. Update your Raspberry Pi](#step_1)
You will need a Raspberry Pi 4 (64-bit ARM processor) running Raspian.  

First, update your Raspberry Pi. Especially if it is your first time booting or you have no used it in a while.  
```bash
sudo apt-get update  
sudo apt-get upgrade  
```  
## [Step 2. Install virtualenv](#step_2) 
Using a virual environment is recommended when developing multiple python projects on the same machine. If you have not done so yet, install a python virtual environment:  
```bash
pip3 install virtualenv
```
If you do not want a virtual environment, you may skip this step and skip any future steps that ask you to activate the environment.  
## [Step 3. Get this repository and start a virtual environment](#step_3)
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
## [Step 4. Setting up the dependencies](#step_4)

### [Option A. Without TPU](#step_4a)
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
  
### [Option B. With TPU](#step_4b)
If you have the Edge TPU and would like the set up script to set it up for you, simply use this command:
```bash
python3 setup.py --setup_TPU=True
```
  
`setup.py` follows the commands listed on [Coral.ai's website](https://coral.ai/docs/accelerator/get-started) to set up the TPU. 

it installed the Edge TPu runtime `libedgetpu1-std` where the std stands for standard clock speed. 

### [Option C. With TPU operating at 2-times default frequency](#step_4c)
Google Coral.ai's Edge TPU can run at the maximum operating frequency. The maximum clock frequency is 2 times the default. *Please note that that when you do this, the Edge TPU will increase in power consumption. This also leads to the Edge TPU possibly becoming very hot.*

To set up the TPU with a higher clock frequency, you can use the following option for setting up:
```bash
python3 setup.py --setup_TPU=True --overclock_TPU=True
```

This will install the edge TPU runtime `libedgetpu1-max`. **You can not have both `libedgetpu1-std` and `libedgetpu1-max` installed at the same time.** If you set it up in either mode, and you would like to switch. you *must* run the `setup.py` again with the desire arguments.