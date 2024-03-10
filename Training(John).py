import numpy as np
import os
import PIL
import tensorflow as tf

import pathlib
import xml.etree.ElementTree as ET

from PIL import Image
import matplotlib.pyplot as plt

user = "John"
normalize = False

'''
Given a directory with an xml files corresponding to images in the signs folder
 open the xml as an image crop the sign based on the bounding box
 save that image in a new folder labeled cropped
'''

if user =="John":
  xml_path = r"C:\Users\johnn\Desktop\Signs\Signs_labeled"
  img_path = r"C:\Users\johnn\Desktop\Signs\Signs"
  cropped_path = r"C:\Users\johnn\Desktop\Signs\Signs_cropped"
  training_path = r"C:\Users\johnn\Desktop\Signs\Training_cropped"

if user == "Frank":
  xml_path = r'''Put the path to the xml files here'''
  img_path = r'''Put the path to the images folder here'''
  cropped_path = r'''Put the path to the cropped folder here'''
  training_path = r'''Put the path to the training folder here'''

for xml in os.listdir(xml_path):
  if not xml.endswith('.xml'): continue
  tree = ET.parse(os.path.join(xml_path, xml))
  root = tree.getroot()
  path = root.find("./path").text
  path = os.path.join(img_path, os.path.basename(path))
  left = float(root.find(".//xmin").text)
  top = float(root.find(".//ymin").text)
  right = float(root.find(".//xmax").text)
  bottom = float(root.find(".//ymax").text)
  print("XML path is ", path)

  if normalize == True:
    img = Image.open(path)
    #Normalize
    img = tf.image.per_image_standardization(img)
    #Save as a jpeg even if the file name doesn't say so
    #img_res.save(os.path.join(cropped_path, os.path.basename(path)), "JPEG")
    imgPath = os.path.join(cropped_path, os.path.basename(path))
    imgByte = tf.image.encode_jpeg(tf.cast(img, tf.uint8))
    tf.io.write_file(imgPath, imgByte)
  
  img = Image.open(path)
  #Crop
  img_res = img.crop((left, top, right, bottom))
  #Convert
  img_res = img_res.convert("RGB")
  img_res.save(path)

'''
   Begin Franks code for trainning
'''
# Load data
data_dir = pathlib.Path(training_path)
image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

#set image heigths
batch_size = 32
img_height = 180
img_width = 180

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


"""   # return if necessary
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

num_classes = 4

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

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)



list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
buffer_size = max(image_count, 1)  # Ensure buffer_size is at least 1
list_ds = list_ds.shuffle(buffer_size, reshuffle_each_iteration=False)
#list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
  print(f.numpy())

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

tf.keras.model.save_weights

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())