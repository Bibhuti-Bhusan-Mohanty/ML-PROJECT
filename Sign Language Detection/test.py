import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Load the model using TensorFlow 2.4.2
model_path ="Model2\keras_model.h5"
labels_path = "Model2\labels.txt"

# TensorFlow 2.4.2 model loading
model = tf.keras.models.load_model(model_path)

# Read the labels from the labels.txt file
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

detector = HandDetector(maxHands=1)

# Initialize parameters
offset = 20
imgSize = 300
counter = 0

while True:
    success, img = cv2.VideoCapture(0).read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white canvas to overlay the resized image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        # Resize image according to aspect ratio
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Normalize image and add batch dimension
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
        imgWhite = imgWhite / 255.0  # Normalize to [0, 1]

        # Predict with TensorFlow model
        predictions = model.predict(imgWhite)
        index = np.argmax(predictions)  # Get the index of the max prediction

        # Draw bounding box and label
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Display cropped images and the processed frame
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
