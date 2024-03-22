import numpy as np
import os
import PIL
import tensorflow as tf
import cv2 as cv

import pathlib
import xml.etree.ElementTree as ET

from PIL import Image
from random import randint
import matplotlib.pyplot as plt

user = "Convert"
normalize = True
crop_images = False


if user =="John":
  xml_path = r"C:\Users\johnn\Desktop\Signs\Sorted\Test"
  img_path = r"C:\Users\johnn\Desktop\Signs\Signs"
  cropped_path = r"C:\Users\johnn\Desktop\Signs\Signs_cropped"
  #training_path = r"C:\Users\johnn\Desktop\Signs\Training_cropped"
  training_path = r"C:\Users\johnn\Desktop\Signs\Training"

if user == "Frank":
  xml_path = r'''Put the path to the xml files here'''
  img_path = r'''Put the path to the images folder here'''
  cropped_path = r'''Put the path to the cropped folder here'''
  training_path = r'''Put the path to the training folder here'''

if user == "Convert":
  xml_path = r"C:\Users\johnn\Desktop\Signs\Sorted\Train"



'''
Given a directory with an xml files corresponding to images in the signs folder
open the xml as an image crop the sign based on the bounding box
save that image in a new folder labeled cropped
'''
for xml in os.listdir(xml_path):
  if not xml.endswith('.xml'): continue
  tree = ET.parse(os.path.join(xml_path, xml))
  root = tree.getroot()
  path = root.find("./path").text
  path = os.path.join(xml_path, os.path.basename(path))
  left = float(root.find(".//xmin").text)
  top = float(root.find(".//ymin").text)
  right = float(root.find(".//xmax").text)
  bottom = float(root.find(".//ymax").text)
  print("XML path is ", path, " File path is ", os.path.join(xml_path, xml))

  if normalize == True:
    img = Image.open(path)
    #convert to RGB
    img = img.convert("RGB")
    #Normalize
    img = tf.image.per_image_standardization(img)
    #Save as a png even if the file name doesn't say so
    if user == "Convert":
      imgPath = os.path.join(xml_path, os.path.basename(path))
    else:
      imgPath = os.path.join(cropped_path, os.path.basename(path))
    imgByte = tf.image.encode_png(tf.cast(img, tf.uint8))
    tf.io.write_file(imgPath, imgByte)

  if crop_images == True:
    img = Image.open(path)
    #Crop
    img_res = img.crop((left, top, right, bottom))
    #Convert
    img_res = img_res.convert("RGB")
    img_res.save(os.path.join(cropped_path, os.path.basename(path)))

exit()
'''
  Begin Franks code for trainning
'''
# Load data
data_dir = pathlib.Path(cropped_path)
print(data_dir)
image_count = len(list(data_dir.glob('*/*.*')))

#set image heigths
batch_size = 4
img_height = 1280
img_width = 720
rgb = 3

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
print("\n")

"""
  # return if necessary
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  # Take the first batch
  for i in range(9):  # Assuming the batch size is at least 9
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))  # Convert tensor to uint8 for visualization
    plt.title(class_names[labels[i].numpy()])  # Ensure label is converted from tensor if necessary
    plt.axis("off")
plt.show()
"""

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Train a model

num_classes = 3

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.summary()

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)



list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
buffer_size = max(image_count, 1)  # Ensure buffer_size is at least 1
list_ds = list_ds.shuffle(buffer_size, reshuffle_each_iteration=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
  print(f.numpy())

#class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
#print(class_names)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

#tf.keras.Model.save_weights
model.save(r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\Fullvision Source Code\Full_Vision\Demo\school.keras")
print('Saving model...')
#tf.saved_model.save(model, r"C:\Users\johnn\Documents\Semester 14 2024 Spring\ECE 397\Fullvision Source Code\Full_Vision\Demo\school")

if test_model == True:
  ### test the model with the video
  image = PIL.Image.open(r"c:\Users\johnn\Desktop\Signs\Test.png")
  image = image.resize((1280, 720), resample=0)
  image = image.convert("RGB")

  image_np = np.array(image)
  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.uint8)

  detection = model(input_tensor)
  print(detection)

  result = {key:value.numpy() for key,value in detection.items()}
  print(result.keys())

  classes = detection['detection_classes'].numpy().astype(int)
  print(classes)
  boxes = detection['detection_boxes'].numpy()
  print(boxes)
  scores = detection['detection_scores'].numpy()
  print(scores)

  for i in range(classes.shape[1]):
      class_id = int(classes[0, i])
      score = scores[0, i]
 
      if np.any(score > 0.5):  # Filter out low-confidence detections
          h, w, _ = image_np.shape
          ymin, xmin, ymax, xmax = boxes[0, i]
 
          # Convert normalized coordinates to image coordinates
          xmin = int(xmin * w)
          xmax = int(xmax * w)
          ymin = int(ymin * h)
          ymax = int(ymax * h)
 
          # Get the class name from the labels list
          class_name = class_names[class_id]
 
          random_color = (randint(0, 256), randint(0, 256), randint(0, 256))
 
          # Draw bounding box and label on the image
          cv.rectangle(image_np, (xmin, ymin), (xmax, ymax), random_color, 2)
          label = f"Class: {class_name}, Score: {score:.2f}"
          cv.putText(image_np, label, (xmin, ymin - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, random_color, 2)
 
  # Display the result
  plt.imshow(image_np)
  plt.axis('off')
  plt.show()

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())