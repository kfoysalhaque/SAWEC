from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8l.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")

#im1 = Image.open("00002.jpg")
#results = model.predict(source=im1, save=True, show=False, save_txt=True, save_conf=True)  # save plotted images

metrics = model.val(data='coco_anechoic.yaml')  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category
