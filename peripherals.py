import cv2
import threading
class Camera:
    def __init__(self, resolution, framerate=30):
        self.stream = cv2.VideoCapture(0) # start the video stream
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        
        # read 1 frame
        (self.grabbed, self.frame) = self.stream.read()
        
        self.stopped = False
    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self 
    
    def update(self):
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                break

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
        
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        
        
