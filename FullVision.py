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

import os                     ### probably will be used for writting to sd card
import cv2 as cv              ### opencv libraries
import numpy as np            ### for image mainpulation "cut out roi"
import threading              ### will be used to multihread video input

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
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.gray = self.frame
        self.stopped = False
        self.stream
        
    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                #self.gray = cv2.cvtColor(self.frame,
                #                         cv2.COLOR_BGR2GRAY)
                
    def stop(self):
        self.stopped = True
      
###############################################################################
###
###   Function Definitions
###
###############################################################################

def SAVE_TO_SD ():
###############################################################################
###
### Given a source copy that source frame to the sd card that will be setup
###   Psuedo Code
###   Check that the external sd card is mounted
###      if yes check that a file is already being written
###         if yes append frame to end of previous file
###            on sucessful writing return
###      if no create a new file with the date (note this will drift)
###   if no sd card is mounted show an alert that there is a problem (LED FLASH)
###
###############################################################################
   pass

def SET_LED(brightness, color):
###############################################################################
###
###   Given a brightness setting and a led color write that data to the led
###
###############################################################################
   pass

def FRAME_MANIP(frame):
###############################################################################
###
###   Given a frame perform the necessary image manipulations
###      This needs to be futher refined before it is written
###
###############################################################################
   pass

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
###   Define variables
###
###############################################################################
CameraSource = 0

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

if __name__ == "__main__": #this will prbly never be called but jic
   video_getter = VideoGet(CameraSource).start()
   frame = video_getter.frame
   grabbed = video_getter.grabbed

   while grabbed == True:
      SAVE_TO_SD(frame)
      mframe = FRAME_MANIP(frame)
      x1,x2,y1,y2 = SEARCH_FRAME(mframe)

# if for some reason the prgram encounters an error and closses or a grabbed frame is missed
# alert the user and write an error to a file on the sd card
###############################################################################
###
###   Begin the clean up operations
###
###############################################################################
WRITE_ERROR(errorcode)
video_getter.stop()