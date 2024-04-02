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

normalize = True
crop_images = False

xml_path = r"c:\Users\johnn\Desktop\Signs\SortedII"

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
        image = Image.open(path)
        image = image.convert('RGB')

        #image = np.array(image)
        #image = image/255.0

        #image = Image.fromarray(image)
        image.save(path, "PNG")