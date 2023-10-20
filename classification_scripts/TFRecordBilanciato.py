# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
import os
from DataAugmentation_Preprocessing import*
import tensorflow as tf
import cv2
import csv
import random
import io
import numpy as np
from Tools import*


modelFile = os.path.dirname(os.path.realpath(__file__)) + "/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = os.path.dirname(os.path.realpath(__file__)) + "/deploy.prototxt"
STANDARD_DIM = (124, 124)
csvFile = 'shuffle_balanced_train.csv'
recordFile = 'shuffle_balanced_train.record'
counter = 0


def cut_face(bigface, image):
    global counter
    x = bigface[0]
    y = bigface[1]
    w = bigface[2]
    h = bigface[3]
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if x < 0 or y < 0 or w < 1 or h < 1 or x > 300 or y > 300:
        return None

    roi_color = image[y:y + h, x:x + w]
    resized = cv2.resize(roi_color, STANDARD_DIM, interpolation=cv2.INTER_AREA)
    return resized


def filtering_faces(faces, image):
    bigface = []
    if len(faces) > 1:
        # print("I'll take the biggest face")
        whmax = 0
        i = 0
        for (x, y, w, h) in faces:
            if w * h > whmax:
                whmax = w * h
                bigface = faces[i].copy()
                i += 1
    if len(faces) == 0:
        return None
    else:
        bigface = faces[0].copy()

    return cut_face(bigface, image)


def preprocessing(image):
    image = sharpen_image(image)
    return image


def detect_image(filename, confidence=0.5):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(filename)
    frameHeight, frameWidth, channels = image.shape
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0), False, False)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    faces = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        conf = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if conf > confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            x1 = int(detections[0, 0, i, 3] * 300)
            y1 = int(detections[0, 0, i, 4] * 300)
            x2 = int(detections[0, 0, i, 5] * 300)
            y2 = int(detections[0, 0, i, 6] * 300)
            f = [x1, y1, x2 - x1, y2 - y1]
            faces.append(f)
    img = filtering_faces(faces, cv2.resize(image, (300, 300)))
    return img


def create_tfrecord():
    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
        with open(csvFile, newline='') as csvfile:
            reader_file = csv.reader(csvfile, delimiter=',')
            for row in reader_file:
                label = round(float(row[1]))
                filename = raw_dataset_path + row[0]
                encoded_jpg = detect_image(filename)
                if encoded_jpg is not None:
                    preprocessed_jpg = preprocessing(encoded_jpg)
                    write_on_tf(preprocessed_jpg,label,writer)
                    if label == 68:
                        if random.randint(0, 1):
                            basic_augmentation(preprocessed_jpg, writer, filename, label)
                    if label in (14, 15, 69, 70):
                        basic_augmentation(preprocessed_jpg, writer, filename, label)
                    if label in (13, 71, 72):
                        medium_augmentation(preprocessed_jpg, writer, filename, label)
                    if label in (5, 6, 7, 12, 73, 74):
                        high_augmentation(preprocessed_jpg, writer, filename, label)
                    if label in range(0, 5) or label in range(8, 12) or label in range(75, 101):
                        extreme_augmentation(preprocessed_jpg, writer, filename, label)
                    #else:
                        #write_on_tf(preprocessed_jpg, label, writer)


with tf.device('/gpu:0'):
    print('GPU ENABLED')
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    tfrecord_file_name = recordFile
    raw_dataset_path = os.path.dirname(os.path.realpath(__file__)) + '/train/'
    create_tfrecord()
