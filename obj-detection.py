import os
import sys
import time
import argparse
import cv2 
import numpy as np
import importlib.util
from threading import Thread

from peripherals import Camera # abstraction for camera module
from model import model # abstraction for tflite model

# some constants that will be used later to normalize the RGB frames (if we use a non-quantized model)
RGB_INPUT_MEAN = 255/2
RGB_INPUT_STD = 255/2

# some display constants and colors
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0 ,255)
BLUE = (255, 0, 0)
LT_GREEN = (10, 255, 0)
LT_BLUE = (255, 255, 0)

# add arguments for flexibility when running program
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", default="./MobileNet_quant/")
arg_parser.add_argument("--graph", default="MobileNet_quant.tflite")
arg_parser.add_argument("--labels", default="labelmap.txt")
arg_parser.add_argument("--video", default=None)
arg_parser.add_argument("--use_TPU", default=False)
arg_parser.add_argument("--confidence_threshold", default=0.5)
arg_parser.add_argument("--resolution", default="1280 x 720")

# get the arguments - we will be using almost all of these arguments at one point or another
args = arg_parser.parse_args()
model_dir = args.model # model to use 
graph_file = args.graph # graph file - *.tflite extention
labels_file = args.labels # labels for the graph
video = args.video # video to use
use_TPU = args.use_TPU  
conf_threshold = float(args.confidence_threshold) # minimum threshold to draw box
resolution = args.resolution # resolution of the camera

if use_TPU:
    model_dir = "./MobileNet_quant_edgeTPU/"
    graph_file = "MobileNet_quant_edgeTPU.tflite"
cwd = os.getcwd() # current working directory

# the model class is an abstraction of the agrument processing - check model.py for details 
# declare the tflite model using the specified model-directory and read the contents
tf_model = model(model_dir)
tf_model.read_model(graph_file, labels_file)
    
if video is not None:
    # if there is a video path, then that takes priority
    video_file = os.path.join(cwd, video)

# next we need to get the target resolution
# first strip any starting/trailing white spaces, then remove the white spaces in between
# next, split the values by the "x" - example 1920x1080 = 1920 width and 1080 height
# finally, make sure they are integer types
resolution = resolution.strip()
resolution = resolution.replace(" " , "")
res_width, res_height = resolution.split('x')
res_width, res_height = int(res_width), int(res_height)
resolution = (res_width, res_height)

framerate_calculation = 1 
clock_freq = cv2.getTickFrequency()

# Get the tensorflow-lite interpreter. There are 2 options here depending on which was installed
# there is a runtime tf-lite interpreter and a python3 tf-lite interpreter, we wil include both in case
# the runtime-tflite is (to my knowledge) faster
# depending on if we are using the TPU or not, we will need to issue different commands to start
# the interpreter 
tf_pkg = importlib.util.find_spec('tflite_runtime')
if use_TPU:
    if tf_pkg is None:
        from tensorflow.lite.python.interpreter import Interpreter
        from tensorflow.lite.python.interpreter import load_delegate
    else:
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
        
    interpreter = Interpreter(model_path=tf_model.graph,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    if tf_pkg is None:
        from tensorflow.lite.python.interpreter import Interpreter
    else:
        from tflite_runtime.interpreter import Interpreter
        
    interpreter = Interpreter(model_path=tf_model.graph)

interpreter.allocate_tensors()

# Get model details
input_tensor_details = interpreter.get_input_details()
output_tensor_details = interpreter.get_output_details()
height = input_tensor_details[0]['shape'][1]
width = input_tensor_details[0]['shape'][2]

# since I'm not sure if the quantization uses int8,16, 32, or 64, I will assume that if the model isnt float32 values, the it is quantized (for now)
quantized_model = not (input_tensor_details[0]['dtype'] == np.float32)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# to allow the user to quit at any time with the 'esc' key 
quit = False 

# now that we have all the information about the model, we will check to see if we want to use a video or a real-time camera
# if a video file was not specified, then we will be using the camera 
if video is None:
    print("Starting Raspi Object detector with the camera. No video file was specified. Press q to quit")
    print(f"Camera resolution is set to: {res_height} x {res_width}")
    

    # Initialize video stream
    camera = Camera(resolution=(res_width,res_height), framerate=30).start()
    time.sleep(1)
   
    while not camera.stopped:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        camera_frame = camera.read()

        # Acquire frame and resize to expected model shape 
        frame = camera_frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # If the model is not quantized, then we need to normalize the pixel values between 0 and 1
        if not quantized_model:
            input_data = (np.float32(input_data) - RGB_INPUT_MEAN) / RGB_INPUT_STD

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_tensor_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_tensor_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_tensor_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_tensor_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_tensor_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                ymin = int(max(1,(boxes[i][0] * res_height)))
                xmin = int(max(1,(boxes[i][1] * res_width)))
                ymax = int(min(res_height,(boxes[i][2] * res_height)))
                xmax = int(min(res_width,(boxes[i][3] * res_width)))
                
                # get the name of the object
                object_name = tf_model.labels[int(classes[i])] # Look up object name from "labels" array using class index
                
                # draw a bounding box around the object
                # openCV take a tuple for the color as the fourth input to the function. but it is NOT In RGB format - instead it is in BGR format
                if object_name == "person":
                    # red box around people
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                elif object_name == "car":
                    # ---- box aound cars
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 0), 2)
                else:
                    # green box around other objects detected
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 0), 2)

                # Draw label
                obj_label = f"{object_name} : {round(scores[i]*100, 2)}%" # put a label with the object name and confidence score on the top right of the box
                labelSize, baseLine = cv2.getTextSize(obj_label, FONT, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
                cv2.putText(frame, obj_label, (xmin, label_ymin-7), FONT, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw framerate in corner of frame
        FPS = f"FPS: {round(frame_rate_calc, 2)}"
        cv2.putText(frame, FPS, (30, 50), FONT, 2, LT_BLUE, 2, cv2.LINE_AA)

        # Finally, project the image with imshow 
        cv2.imshow('Real-Time Object detector with TF-LITE', frame)
        
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'esc' to quit at any time
        key = cv2.waitKey(1)
        if key == 27:
            print("Quitting raspberry pi object detector!")
            camera.stop()


    cv2.destroyAllWindows()
else:
    # a video path was specified, as such we will perform object detection on the video instead
    # Open video file
    video = cv2.VideoCapture(video_file)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    while(video.isOpened() and not quit):
        t1 = cv2.getTickCount()
        # Acquire frame and resize to expected shape [1xHxWx3]
        next_frame_exists, frame = video.read()
        if not next_frame_exists:
            print("Reached the end of the video, Quitting raspberry pi object detector")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # If the model is not quantized, then we need to normalize the pixel values between 0 and 1
        if not quantized_model:
            input_data = (np.float32(input_data) - RGB_INPUT_MEAN) / RGB_INPUT_STD

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_tensor_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_tensor_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_tensor_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_tensor_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_tensor_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # get the name of the object
                object_name = tf_model.labels[int(classes[i])] # Look up object name from "labels" array using class index
                
                # draw a bounding box around the object
                # openCV take a tuple for the color as the fourth input to the function. but it is NOT In RGB format - instead it is in BGR format
                if object_name == "person":
                    # red box around people
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 1)
                elif object_name == "car":
                    # ---- box aound cars
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 0), 1)
                else:
                    # green box around other objects detected
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 0), 1)
                    pass

                # Draw label
                obj_label = f"{object_name} : {round(scores[i]*100, 2)}%" # put a label with the object name and confidence score on the top right of the box
                labelSize, baseLine = cv2.getTextSize(obj_label, FONT, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-12), (255, 255, 255), cv2.FILLED) 
                cv2.putText(frame, obj_label, (xmin, label_ymin-7), FONT, 0.5, (0, 0, 0), 1) # Draw label text

        # Draw framerate in corner of frame
        FPS = f"FPS: {round(frame_rate_calc, 2)}"
        cv2.putText(frame, FPS, (30, 50), FONT, 1, RED, 2, cv2.LINE_AA)
        if use_TPU:
            cv2.putText(frame, "USING EDGE TPU", (10, height - 10), FONT, 0.5, RED, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "WITHOUT EDGE TPU", (10, height - 10), FONT, 0.5, RED, 2, cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Kelvin Ma - Object Detector', frame)
        
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'esc' to quit at any time
        key = cv2.waitKey(1)
        if key == 27:
            quit = True

    # Clean up
    video.release()


