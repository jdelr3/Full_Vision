import tensorflow as tf
import keras_cv
import keras
import numpy as np
from keras_cv import bounding_box
from keras_cv import visualization

model_path = r"c:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\model.keras"
input_path = r"c:\Users\johnn\Pictures\Screenshots\Screenshot (84).png"

class_ids = [
    "END",
    "SCHOOL",
    "PEDESTRIAN",
]

class_mapping = dict(zip(range(len(class_ids)), class_ids))

image = keras.utils.load_img(input_path)
image = np.array(image)

visualization.plot_image_gallery(
    np.array([image]),
    value_range=(0, 255),
    rows=1,
    cols=1,
    scale=5,
)

inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
)

image_batch = inference_resizing([image])

model = keras.models.load_model(model_path)



y_pred = model.predict(image_batch)
print(y_pred)
# y_pred is a bounding box Tensor:
# {"classes": ..., boxes": ...}
#visualization.plot_bounding_box_gallery(
#    np.array(image_batch),
#    value_range=(0, 255),
#    rows=1,
#    cols=1,
#    y_pred=y_pred,
#    scale=5,
#    font_scale=0.7,
#    bounding_box_format="xyxy",
#    class_mapping=class_mapping,
#)