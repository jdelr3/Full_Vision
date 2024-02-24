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

import os            ### probably will be used for writting to sd card
import cv2 as cv     ### opencv libraries
import numpy as np   ### for image mainpulation "cut out roi"
import threading     ### will be used to multihread video input
import datetime      ### used to get the date in YYYY-MM-DD for folder
import argparse		### implementation of command line flags to control features

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
   

def SET_LED(brightness, color):
###############################################################################
###
###   Given a brightness setting and a led color write that data to the led
###
###############################################################################
   pass

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

###############################################################################
###
###   Argrument parsing
###
###############################################################################
parser = argparse.ArgumentParser(description="Gives the ability to enable and \
   disable features as necessary during the programming testing phase")
parser.add_argument('-S',"--ShowVideo", action="store_true", 
                    help="Show video on desktop")
parser.add_argument('-M', "--Demo", action="store_true", 
                    help="Use a video from  the desktop")
parser.add_argument('-D', "--Debug", action="store_true")

args = parser.parse_args()

###############################################################################
###
###   Define variables
###
###############################################################################
if args.Demo:
   CameraSource = '/home/fullvison/Full_Vision/Fullerton_720.mp4'
else:
   CameraSource = 0

errorcode = 0
path_to_SD = "/media/externalSD/" ## This is an example will change once the
                                 #on the hardware and confirm directory


###############################################################################
###
###   Error table
###
###############################################################################
noSD = -1      # error code 1 no external sd mounted
msngFrme = -2  # error code 2 missing or corrupted frame

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

if __name__ == "__main__": #this will probably never be called but jic
#   video_getter = VideoGet(CameraSource).start()
#   frame = video_getter.frame
#   grabbed = video_getter.grabbed
   srcVideo = cv.VideoCapture(CameraSource)
   grabbed, frame = srcVideo.read()

   ### setup the opencv video writer
   saveName = GET_SAVE_FILE(path_to_SD)
   print(saveName)
   fourcc = cv.VideoWriter_fourcc(*'mp4v') 
   #out = cv.VideoWriter(saveName + "test.mp4", fourcc, 30, (1920,1080), 0)
   
   while grabbed == True and errorcode == 0:
#      frame = video_getter.frame
#      grabbed = video_getter.grabbed
      grabbed, frame = srcVideo.read()
      
      if args.ShowVideo:
         cv.imshow("TestOut",frame)
         cv.waitKey(1)
      #if SAVE_TO_SD(saveName, frame, out) != -1:
      #   mframe = FRAME_MANIP(frame)
      #   x1,x2,y1,y2 = SEARCH_FRAME(mframe)
      #else:
      #   errorcode = noSD

   ### if the while loop is broken check that the frame isn't missing   
   if grabbed == False:
      errorcode = msngFrme
      print("error missing frame") 
      
# if for some reason the prgram encounters an error and closses or a grabbed frame is missed
# alert the user and write an error to a file on the sd card
###############################################################################
###
###   Begin the clean up operations
###
###############################################################################
WRITE_ERROR(errorcode)
#video_getter.stop()