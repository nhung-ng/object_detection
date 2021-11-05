import numpy as np
import time
import cv2
import os

confthres=0.25
nmsthres=0.4 # IoU between two predictions BBoxes
yolo_path="./"

def get_labels(labels_path):
    # load the class labels 
    #labelsPath = os.path.sep.join([yolo_path, "data/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # init a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # the path to the  weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath) #read configuration of yolo
    return net


