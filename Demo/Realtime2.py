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
input_path = r"c:\Users\johnn\Pictures\Screenshots\Screenshot (415).png"
output_path = r"c:\Users\johnn\Pictures\Screenshots\Screenshot.png"
video_path = r"c:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\FullertonAve.mp4"

maxwidth, maxheight = 640, 640

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

THRESHOLD = .1

class_ids = [
    "none",
    "END",
    "SCHOOL",
    "PEDESTRIAN",
]

class_mapping = dict(zip(range(len(class_ids)), class_ids))

if from_vid == True:
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()

        #cv.imshow("show im", frame)
        #cv.imshow("Test", frame)
        #cv.waitKey(1)

        if model_use == "keras":
            f1 = maxwidth / frame.shape[1]
            f2 = maxheight / frame.shape[0]
            f = min(f1, f2)
            dim = (int(frame.shape[1] * f), int(frame.shape[0] * f))
            rframe = cv.resize(frame, dim)
            h, w = rframe.shape[:2]
            if h < w:
                top = (maxheight - h) / 2
                bottom = (maxheight - h) / 2
                left = 0
                right = 0
            else:
                top = 0
                bottom = 0
                left = (maxwidth - w) / 2
                right = (maxwidth - w) / 2
            rframe = cv.copyMakeBorder(rframe, int(top), int(bottom), int(left), int(right), cv.BORDER_CONSTANT)
        else:
            rframe = cv.resize(frame, (maxwidth,maxheight))
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
                            x_min = x_min
                            y_min = y_min
                            x_max = x_max
                            y_max = y_max
                            color = (0, 255, 0)  # Green color (BGR format)
                            thickness = 2
                            cv.rectangle(rframe, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        

        else:
            confidence = np.array(results[confidence_scr])
            bbox = np.array(results[boxes])

            for signIndex, sign in enumerate(confidence):
                #print(sign)
                #print(classes[signIndex])
                if sign.any() > THRESHOLD * 100:
                    for sIndex, s in enumerate(sign):
                        #print(sIndex, " ", s)
                        if s.any() > THRESHOLD * 100:
                            for cIndex, c in enumerate(s):
                                if c > THRESHOLD * 100:
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
        
        cv.imshow('YOLO', rframe)     
        if cv.waitKey(1) & 0xFF == ord(' '):
            exit()    
    cap.release()

if from_vid ==  "test_bb":
    frame = cv.imread(input_path, 1)
    
    if model_use == "keras":
        f1 = maxwidth / frame.shape[1]
        f2 = maxheight / frame.shape[0]
        f = min(f1, f2)
        dim = (int(frame.shape[1] * f), int(frame.shape[0] * f))
        rframe = cv.resize(frame, dim)
        h, w = rframe.shape[:2]
        if h < w:
            top = (maxheight - h) / 2
            bottom = (maxheight - h) / 2
            left = 0
            right = 0
        else:
            top = 0
            bottom = 0
            left = (maxwidth - w) / 2
            right = (maxwidth - w) / 2
        rframe = cv.copyMakeBorder(rframe, int(top), int(bottom), int(left), int(right), cv.BORDER_CONSTANT)
    else:
        rframe = cv.resize(frame, (maxheight,maxwidth))
    #cv.imshow("test",rframe)
    nframe = np.array(rframe)
    nframe = np.expand_dims(nframe, 0)
    nframe = nframe/255.0

    results = model(nframe)
    print(results)
    #print(results['detection_boxes'])
    keras_cv.visualization.draw_bounding_boxes(
            nframe,
            results[boxes],
            bounding_box_format='xyxy',
            class_mapping=class_mapping,
            color=(255,0,0)
        )
    cv.imwrite(output_path, rframe)


else:
    frame = cv.imread(input_path, 1)
    
    if model_use == "keras":
        f1 = maxwidth / frame.shape[1]
        f2 = maxheight / frame.shape[0]
        f = min(f1, f2)
        dim = (int(frame.shape[1] * f), int(frame.shape[0] * f))
        rframe = cv.resize(frame, dim)
        h, w = rframe.shape[:2]
        if h < w:
            top = (maxheight - h) / 2
            bottom = (maxheight - h) / 2
            left = 0
            right = 0
        else:
            top = 0
            bottom = 0
            left = (maxwidth - w) / 2
            right = (maxwidth - w) / 2
        rframe = cv.copyMakeBorder(rframe, int(top), int(bottom), int(left), int(right), cv.BORDER_CONSTANT)
        #rframe = cv.resize(frame, (320,320))
        #rframe = tf.image.resize(frame, (320,320),preserve_aspect_ratio=True)
    else:
        rframe = cv.resize(frame, (maxheight,maxwidth))
    #cv.imshow("test",rframe)
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
                        x_min = x_min
                        y_min = y_min
                        x_max = x_max
                        y_max = y_max
                        color = (0, 255, 0)  # Green color (BGR format)
                        thickness = 2
                        cv.rectangle(rframe, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        
        cv.imwrite(output_path, rframe)
        
    else:
        confidence = np.array(results[confidence_scr])
        bbox = np.array(results[boxes])

        for signIndex, sign in enumerate(confidence):
            #print(sign)
            #print(classes[signIndex])
            if sign.any() > THRESHOLD * 100:
                for sIndex, s in enumerate(sign):
                    #print(sIndex, " ", s)
                    if s.any() > THRESHOLD * 100:
                        for cIndex, c in enumerate(s):
                            if c > THRESHOLD * 100 and cIndex != 0:
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
        
        cv.imwrite(output_path, rframe)

cv.destroyAllWindows()