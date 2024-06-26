#! /usr/bin/python3

################################################################################
###
###     Senior Design FullVision
###      Borrowed from the sample codes written week 2
###      Week 3 Create a foundation for the FullVision software
###         Create skeleton functions, begin buliding classes? psuedoocode things
###
################################################################################

################################################################################
###
###     Import Libraries
###
################################################################################

import argparse		### implementation of command line flags to control features
import time         ### for calculating time delays
from W2812 import SPItoWS ### LED controller

import os            ### used for writting to sd card
import datetime      ### used to get the date in YYYY-MM-DD for folder

import tensorflow as tf ### load the keras model
from tensorflow import keras ### high level cv library
import cv2 as cv     ### opencv libraries
import numpy as np   ### for image mainpulation "cut out roi"

### Possible deletion depending on speed
import threading     ### will be used to multihread video input

################################################################################
###
###     Class Definitions
###
################################################################################

class VideoGet:
################################################################################
###
###   This was a video get class I wrote for a different much slower processor
###      will definitely need to be reworked with the increase in processing 
###      power from the arm cpu
###   Intialize the class and create the class variables
###   start class function
###      Begin threading the classes get function
###   get class function
###      if the program is running read the frame from a predefined camera source
###      Else stop
###   stop class function
###      This should never be used but cleans up the for the class and ends programs
###      this should be removed or given an error producing code
###
#################################################################################
    def __init__(self,src):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.gray = self.frame
        self.stopped = False
        self.stream
        
    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                
    def stop(self):
        self.stopped = True
      
###############################################################################
###
###   Function Definitions
###
###############################################################################

def GET_SAVE_FILE(base_path):
###############################################################################
###
###   Given a path to the external sd card
###   Check if a folder already exists for this date, if no create
###    create a new filename with the current time (this will be on startup)
###   return the file path   
###
###############################################################################
   if os.path.isdir(base_path):
      os.chdir(base_path)
      daySTR = datetime.date.today().isoformat()
      if os.path.isdir(base_path + daySTR) != True:
         os.mkdir(base_path + daySTR)
      return str(base_path + daySTR)   #return as a string
   else:
      return -1                        #return -1 no SD Card mounted

def SAVE_TO_SD (file_path, frame, writer):
###############################################################################
###
### Given a source copy that source frame to the sd card that will be setup
###   Psuedo Code
###   xCheck that the external sd card is mounted
###     -if yes check that a file is already being written (taken care of above)
###         xif yes append frame to end of previous file
###            xon sucessful writing return
###      -if no create a new file with the date (note this will drift) (taken care of above)
###   xif no sd card is mounted show an alert that there is a problem (LED FLASH)
###
###############################################################################
   if os.path.isdir(file_path):
      writer.write(frame)
   else:
      return -1
   
def GET_BRIGHTNESS(sign):
    divVal = max(10-bValue, 1) ### prevent divide by zero
    sign = (sign[0], int(sign[1]/divVal), int(sign[2]/divVal), int(sign[3]/divVal))
    return sign

def ERROR_LED (errorCode):
###############################################################################
###
### Given an error code flash the light red that many times followed by a
###     half second delay since error this is a trap condition
###
###############################################################################
    i = -errorcode ## invert error code to positive value
    print("Encountered error: ", errorCode)
    dimError = GET_BRIGHTNESS(errColor)
    while True:
        while i > 0:
            WS.RGBto3Bytes(dimError[0], dimError[1], dimError[2], dimError[3])
            WS.LED_show()
            time.sleep(0.25)
            WS.LED_OFF_ALL()
        time.sleep(.75)   

def SET_LED(sign):
###############################################################################
###
###   Given an led color write that data to the led
###     return time led turned on for timeout
###
###############################################################################
    dimSign = GET_BRIGHTNESS(sign)
    WS.RGBto3Bytes(dimSign[0], dimSign[1], dimSign[2], dimSign[3])
    WS.LED_show()
    return time.time_ns()

'''
def preprocess(frame):
###############################################################################
###
### Given the frame convert to HSV, threshold with same values as model trained
###     on return an binary image
###
###############################################################################
   
   #rows, cols, null = frame.shape
   #imgHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
   #BMIN = np.array([100, 43, 46])
   #BMAX = np.array([124,255,255])
   #BMIN = np.array([50,25,20])
   #BMAX = np.array([160,255,255])
   #img_Bbin = cv.inRange(imgHSV, BMIN, BMAX)
   
   #Rmin1 = np.array([0, 43, 46])
   #Rmax1 = np.array([10, 255, 255])
   #Rmin1 = np.array([0, 60, 60])
   #Rmax1 = np.array([10, 255, 255])
   #img_Rbin1 = cv.inRange(imgHSV, Rmin1, Rmax1)

   #Rmin2 = np.array([156, 43, 46])
   #Rmax2 = np.array([180, 255, 255])
   #img_Rbin2 = cv.inRange(imgHSV, Rmin2, Rmax2)
   #img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
   #img_bin = np.maximum(img_Bbin, img_Rbin)

   imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
   thresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
   return thresh
   #return img_bin

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv.findContours(img_bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

def getClassName(classNo):
    if classNo == 9:
        return 'No passing'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Wild animals crossing'
    else:
        return ""
        
def PIXEL_NORMALIZATION(frame):
###############################################################################
###
###   Normalize video input to correct contrast and make picture easier to
###      operate on, also resize to 720
###
###############################################################################
   alpha = 0
   beta = 0
   frame_720 = np.zeros((1280,720))
   return cv.normalize(frame,frame_720,alpha,beta,cv.NORM_MINMAX)

def FRAME_MANIP(frame):
###############################################################################
###
###   Given a frame perform the necessary image manipulations
###      This needs to be futher refined before it is written
###
###############################################################################
   if frame is not None:
      ## frank put function call here
      frame = PIXEL_NORMALIZATION(frame)

def SEARCH_FRAME(frame):
###############################################################################
###
###   Given a previoulsy maniplated frame search for roi
###      for this early search use something very quick
###   if a sign is found return the x1,x2,y1,y2 coordinates od the ROI
###   if nothing return a -1, -1, -1, -1
###
###############################################################################
   pass

def WRITE_ERROR(errorcode):
###############################################################################
###
###   Given an errorcode write a file to the sd card explaining what the error
###   is or most likely is
### 
###############################################################################
   pass
'''

###############################################################################
###
###   Argrument parsing
###
###############################################################################
parser = argparse.ArgumentParser(description="Gives the ability to enable and \
   disable features as necessary during the programming testing phase")
parser.add_argument('-S',"--ShowVideo", action="store_true", 
                    help="Show video on desktop")
parser.add_argument('-I', "--input", type=str,
                    help="Use a video from  the desktop")
parser.add_argument('-V', "--verbose", action="store_true", 
                    help="ouput debug info")
parser.add_argument('-M', "--model", type=str,
                    help="model file to load")
parser.add_argument('-T', "--treshhold", type=float,
                    help="Changes the threshold value")
parser.add_argument('-B', "--brightness", type=int,
                    help="value from 1-10 brightness 10 is max")

args = parser.parse_args()

###############################################################################
###
###   Define variables
###
###############################################################################

#model threshold
threshold = args.threshold
font = cv.FONT_HERSHEY_COMPLEX
maxwidth, maxheight = 320, 320 # size for dectector

#Video variables
if args.input !=None:
   CameraSource = args.input
   #CameraSource = r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\FullertonAve.mp4"
   #CameraSource = r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\Frankfurt.mp4"
else:
   ### load the camera using the nvidia camera loader with max input size and
   ###  and output size of 960x616. requires less resources than loading dev/video/0 and
   ###  resizing manually
   CameraSource = "nvarguscamerasrc sensor_mode=0 ! video/x-raw(memory:NVMM), width=(int)3820, height=(int)2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, width=(int)960, height=(int)616, format=(string)BGRx ! videoconvert ! appsink"

#error codes
errorcode = 0 #initalize error code value, red output flashing code errors
path_to_SD = "/media/externalSD/" ## This is an example will change once the
                                 #on the hardware and confirm directory

#LED setup
WS = SPItoWS(1)
WS.LED_OFF_ALL() 
schoolColor = (0, 255, 216, 0)   ### RGB color values of sign 
pedColor = (0, 247, 181, 0)      ### RGB color values of sign
errColor = (0, 255, 0, 0)        ### RGB value error

bValue = args.brightness


###############################################################################
###
###   Error table
###
###############################################################################
noSD = -1      # error code 1 no external sd mounted
msngFrme = -2  # error code 2 missing or corrupted frame
noThresh = -3  # error code 3 missing threshhold infomration
noModel = -4   # error code 4 no model information 

###############################################################################
###
###   Main loop
###   Get frame from video getter class
###   check to make sure that the frame has been grabbed
###   Write the frame to the SD card
###   Image manipulation to allow for easier sign detection
###   Search in frame for signs
###   if a sign is found Extract roi
###      Classify sign 
###      if no classificaiton can be found save coordinates to check in next frame
###      if classification has been made add to count if x same classifications set alert
###         after y amount of time clear the count
###     
############################################################################### 

print(args)
#
#if __name__ == "__main__": #this will probably never be called but jic
##    
##    load the model shamefully taken from https://github.com/TotallyStud/live-road-sign-detection/tree/main
##    This group trained this model on the German Traffic Sign Recognition dataset. for the final project we
##    will have developed our own model but for the demonstration this will do
##    
#    model = keras.models.load_model(r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\model.keras")
#
#    # grab the first frame and then just begin the detection
#    srcVideo = cv.VideoCapture(CameraSource)
#    grabbed, frame = srcVideo.read()
#
#    cols = 1280
#    rows = 720
#
#    print(cols, rows)
#    ### setup the opencv video writer
#    #saveName = GET_SAVE_FILE(path_to_SD)
#    #print(saveName)
#    #fourcc = cv.VideoWriter_fourcc(*'mp4v') 
#    #out = cv.VideoWriter(saveName + "test.mp4", fourcc, 30, (1920,1080), 0)
#   
#    while grabbed == True and errorcode == 0:
##      frame = video_getter.frame
##      grabbed = video_getter.grabbed
#        grabbed, frame = srcVideo.read()
#        
#        #preprocess the video
#        #img_bin = frame.resize((320,320))
#        img_bin = preprocess(frame)
#        min_area = frame.shape[0] * frame.shape[1] / (25 * 25)
#        cv.imshow("processed image", img_bin)
#        
#        #get contours
#        rects = contour_detect(img_bin, min_area)
#
#        # get center and draw bounding boxes
#        for rect in rects:
#            xc = int(rect[0] + rect[2] / 2)
#            yc = int(rect[1] + rect[3] / 2)
#            size = max(rect[2], rect[3])
#            x1 = max(0, int(xc - size / 2))
#            y1 = max(0, int(yc - size / 2))
#            x2 = min(cols, int(xc + size / 2))
#            y2 = min(rows, int(yc + size / 2))
#
#            if rect[2] > 100 and rect[3] > 100:
#                cv.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
#            crop_img = np.asarray(frame[y1:y2, x1:x2])
#            crop_img = cv.resize(crop_img, (32, 32))
#            crop_img = preprocess(crop_img)
#            crop_img = crop_img.reshape(1, 32, 32, 1)
#            predictions = model.predict(crop_img)
#            classIndex = np.argmax(predictions, axis=-1)
#            probabilityValue = np.amax(predictions)
#            if probabilityValue > threshold:
#                if getClassName(classIndex) != "":
#                    cv.putText(frame, str(classIndex) + " " + str(getClassName(classIndex)), (rect[0], rect[1] - 10),
#                            font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
#                    cv.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
#                            (0, 0, 255), 2, cv.LINE_AA)
#        
#        if args.ShowVideo:
#            cv.imshow("Output", frame)  # Display the output
#            cv.waitKey(25)
#            
#        #if SAVE_TO_SD(saveName, frame, out) != -1:
#        #   mframe = FRAME_MANIP(frame)
#        #   x1,x2,y1,y2 = SEARCH_FRAME(mframe)
#        #else:
#        #   errorcode = noSD
#
#    ### if the while loop is broken check that the frame isn't missing   
#    if grabbed == False:
#        errorcode = msngFrme
#        print("error missing frame") 
#      
## if for some reason the prgram encounters an error and closses or a grabbed frame is missed
## alert the user and write an error to a file on the sd card
################################################################################
####
####   Begin the clean up operations
####
################################################################################
#WRITE_ERROR(errorcode)
##video_getter.stop()
#
#'''