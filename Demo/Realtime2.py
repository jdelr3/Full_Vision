import tensorflow as tf
import keras_cv
import keras
import cv2 as cv
import numpy as np

model_use = "keras"
from_vid = False

model_path = r"c:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\model100.keras"
model_pb = r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\saved_model"
model_30_pb = r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\TestModel\saved_model"
input_path = r"c:\Users\johnn\Pictures\Screenshots\Screenshot (284).png"
output_path = r"c:\Users\johnn\Pictures\Screenshots\Screenshot.png"
video_path = r"c:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\FullertonAve.mp4"

if model_use == "keras":
    model = keras.models.load_model(model_path)
    boxes = str('boxes')
    confidence_scr = str('classes')
if model_use == "pb_10":
    model = tf.saved_model.load(model_pb)
    confidence_scr = str('detection_scores')
    bbox = str('detection_boxes')
    classes = str('detection_classes')
if model_use == "pb_30":
    model = tf.saved_model.load(model_30_pb)
    confidence_scr = str('detection_scores')
    bbox = str('detection_boxes')
    classes = str('detection_classes')

THRESHOLD = .10

class_ids = [
    "none",
    "END",
    "SCHOOL",
    "PEDESTRIAN",
]

if from_vid == True:
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()

        #cv.imshow("show im", frame)
        #cv.imshow("Test", frame)
        #cv.waitKey(1)

        if model_use == "keras":
            rframe = cv.resize(frame, (320,320))
        else:
            rframe = cv.resize(frame, (320,320))
        nframe = np.array(rframe)
        nframe = np.expand_dims(nframe, 0)
        nframe = nframe/255.0

        # BGR to RGB conversion is performed under the hood
        # see: https://github.com/ultralytics/ultralytics/issues/2575
        results = model(nframe)
        #print(results)
        #print(results['detection_boxes'])
        if model_use != "keras":
            confidence = np.array(results['detection_scores'])
            bbox = np.array(results['detection_boxes'])
            classes = np.array(results['detection_classes'])

            for signIndex, sign in enumerate(confidence):
                #print(classes)
                #print(classes[signIndex])
                if sign.any() > THRESHOLD:
                    for sIndex, s in enumerate(sign):
                        #print(sIndex, " ", s)
                        if s > THRESHOLD:
                            #print(int(classes[signIndex][sIndex]))
                            print(class_ids[int(classes[signIndex][sIndex])])
                            print(bbox[signIndex][sIndex][0:4])
                            x_min, y_min, x_max, y_max = np.array(bbox[signIndex][sIndex][0:4])
                            x_min = x_min * 320
                            y_min = y_min * 320
                            x_max = x_max * 320
                            y_max = y_max * 320
                            color = (0, 255, 0)  # Green color (BGR format)
                            thickness = 2
                            cv.rectangle(rframe, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        

        else:
            confidence = np.array(results[confidence_scr])
            bbox = np.array(results[boxes])

            for signIndex, sign in enumerate(confidence):
                #print(sign)
                #print(classes[signIndex])
                if sign.any() > THRESHOLD:
                    for sIndex, s in enumerate(sign):
                        #print(sIndex, " ", s)
                        if s.any() > THRESHOLD:
                            for cIndex, c in enumerate(s):
                                if c > THRESHOLD:
                                    print(class_ids[int(cIndex)])
                                    print(bbox[signIndex][sIndex][0:4])
                                    x_min, y_min, x_max, y_max = np.array(bbox[signIndex][sIndex][0:4])
                                    x_min = x_min * 320
                                    y_min = y_min * 320
                                    x_max = x_max * 320
                                    y_max = y_max * 320
                                    color = (0, 255, 0)  # Green color (BGR format)
                                    thickness = 2
                                    cv.rectangle(rframe, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        
        cv.imshow('YOLO', frame)     
        if cv.waitKey(1) & 0xFF == ord(' '):
            exit()    
    cap.release()

else:
    frame = cv.imread(input_path, 1)
    
    if model_use == "keras":
        rframe = cv.resize(frame, (320,320),)
    else:
        rframe = cv.resize(frame, (320,320))
    nframe = np.array(rframe)
    nframe = np.expand_dims(nframe, 0)
    nframe = nframe/255.0

    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model(nframe)
    print(results)
    #print(results['detection_boxes'])
    if model_use != "keras":
        confidence = np.array(results['detection_scores'])
        bbox = np.array(results['detection_boxes'])
        classes = np.array(results['detection_classes'])

        for signIndex, sign in enumerate(confidence):
            #print(classes)
            #print(classes[signIndex])
            if sign.any() > THRESHOLD:
                for sIndex, s in enumerate(sign):
                    #print(sIndex, " ", s)
                    if s > THRESHOLD:
                        #print(int(classes[signIndex][sIndex]))
                        print(class_ids[int(classes[signIndex][sIndex])])
                        print(bbox[signIndex][sIndex][0:4])
                        x_min, y_min, x_max, y_max = np.array(bbox[signIndex][sIndex][0:4])
                        x_min = x_min * 320
                        y_min = y_min * 320
                        x_max = x_max * 320
                        y_max = y_max * 320
                        color = (0, 255, 0)  # Green color (BGR format)
                        thickness = 2
                        cv.rectangle(rframe, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        
        cv.imwrite(output_path, frame)
        
    else:
        confidence = np.array(results[confidence_scr])
        bbox = np.array(results[boxes])

        for signIndex, sign in enumerate(confidence):
            #print(sign)
            #print(classes[signIndex])
            if sign.any() > THRESHOLD:
                for sIndex, s in enumerate(sign):
                    #print(sIndex, " ", s)
                    if s.any() > THRESHOLD:
                        for cIndex, c in enumerate(s):
                            if c > THRESHOLD and cIndex != 0:
                                print(class_ids[int(cIndex)])
                                print(bbox[signIndex][sIndex][0:4])
                                x_min, y_min, x_max, y_max = np.array(bbox[signIndex][sIndex][0:4])
                                x_min = x_min
                                y_min = y_min
                                x_max = x_max
                                y_max = y_max
                                color = (0, 255, 0)  # Green color (BGR format)
                                thickness = 2
                                cv.rectangle(rframe, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        
        cv.imwrite(output_path, frame)

cv.destroyAllWindows()