########################################################################
# Project: Object detection using a RasPi4 and AI Accelerator
# 
# Date: 2020/04/05
# Author: Kelvin Ma
# Description: This file is the set up file to install necessary files
# and packages for TFLite and OpenCV(3.4)
#
# Class: EE45900 - Microprocessors - Spring 2020
# Professor: Prof. Pekcan
# Partner: Younghwa Min
#
# NOTE: this script only has to be run ONCE and only works 
# on ARM PROCESSORS (like the Raspberry Pi 4) on python 3.5, 3.6 or 3.7
# list here: https://www.tensorflow.org/lite/guide/python
########################################################################

import os
import sys

print("Setting up neccessary files for Object Detection")
os.system('date')

# install dependencies for OpenCV v3.4.6.27
cv_version = "3.4.6.27"
openCV_dependencies = [
	"libjpeg-dev", 
	"libtiff5-dev",
	"libjasper-dev",
	"libpng12-dev",
	"libavcodec-dev",
	"libavformat-dev",
	"libswscale-dev",
	"libv4l-dev",
	"libxvidcore-dev",
	"libx264-dev",
	"qt4-dev-tools",
	"libatlas-base-dev",
]

for dependency in openCV_dependencies:
	print(f"installing {dependency}")
	cmd = f"sudo apt-get -y install {dependency}"
	os.system(cmd)

# install OpenCV 3.4.6.27
os.system(f"pip3 install opencv-python=={cv_version}")

# install Tensorflow-Lite 
# Although this code was developed on Python 3.7, tflite also supports
# Python 3.5 and can use the same code. however, they have different 
# python package locations, as such we need to check the version of 
# python installed on the current system
version = sys.version
if version[:3] == "3.7": 
	print(f"Installing TensorFlow-Lite for Python{version[:3]}")
	os.system("pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl")
elif version[:3] == "3.6":
	print(f"Installing TensorFlow-Lite for Python{version[:3]}")
	os.system("pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_armv7l.whl")
	pass
elif version[:3] == "3.5":
	print(f"Installing TensorFlow-Lite for Python{version[:3]}")
	os.system("pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl")
	pass
else:
	# raise an error if the python version is incompatible 
	print("SET UP FAILED")
	raise ValueError("Unsupported Python version. Must be Python 3.5, Python 3.6 or Python 3.7")
