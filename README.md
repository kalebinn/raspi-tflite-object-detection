# Object Detection using a Raspberry Pi 4 and TensorFlow Lite
With optional Google Coral.ai USB AI Accelerator (Edge TPU co-processor)  
  
<img src= "./docs/imgs/dashcam-noTPU.gif" width=300><img src= "./docs/imgs/dashcam-usingTPU.gif" width=300>

## Description  
This is a project for EE 45900 - Microprocessors - Spring 2020 at The City College of New York. In this project we will use a Raspberry Pi to create a Real-Time Object Detector.  
Group members: Kelvin Ma and YoungHwa Min  
  
We will be using a [Raspberry Pi 4 (4 GB) model](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/?variant=raspberry-pi-4-model-b-4gb) and [Coral's USB Accelerator](https://coral.ai/products/accelerator) along with [TensorFlow-Lite](https://www.tensorflow.org/lite/) and [OpenCV](https://opencv.org/). In addition to these things this project features a multi-threaded PiCamera for a small performance boost (in FPS).  
  
This project can be used as an extention for other projects such as smart street light cameras, dash cams for cars, UAV surveilance, etc. As such it was designed to be as expandable as possible.
  
This project will be open source and open to modification. In fact, it was made possible by many examples and open source code. We have provided credit in the code depending on the resources. Our own original code will also be open source to all, we only ask that you provide credit. If there are any questions/mistakes, feel free to submit a pull request for e-mail me directly at kalebinn@gmail.com.  

## Useful Docs
[Set up on your own Raspberry Pi 4 - without the TPU](./docs/set_up_instructions.md)  
[User's Guide](./docs/Users_Guide.md)

## Requirements
In order to run this project, you will need the following:  
* ARM based Raspberry Pi 4 - 4GB model B(other version of Raspberry Pi are untested).
* Raspian Buster or Raspian Stretch OS loaded on the Raspberry Pi
* PiCamera (v1.x or v2 will work), or USB Webcam connect to the Raspberry Pi
* Python 3.5.x, Python 3.6.x or Python 3.7.x
* TensorFlow-Lite interpreter v2.1 
* OpenCV v3.4
