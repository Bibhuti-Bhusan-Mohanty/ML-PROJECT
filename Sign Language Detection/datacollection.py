import cv2  # Import OpenCV library for image processing
from cvzone.HandTrackingModule import HandDetector  # Import HandDetector from cvzone for hand tracking
import numpy as np  # Import NumPy for array manipulations
import math  # Import math library for calculations
import time  # Import time library for timestamping saved images

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Initialize hand detector with a maximum of one hand detected at a time
detector = HandDetector(maxHands=1)

# Define constants
offset = 20  # Padding around the detected hand
imgSize = 300  # Size of the output image
counter = 0  # Counter for saved images

# Folder to save processed images
folder = "Data\Thank You"

# Main loop to capture and process frames
while True:
    success, img = cap.read()  # Capture a frame from the webcam
    hands, img = detector.findHands(img)  # Detect hands in the frame and draw landmarks
    if hands:  # If at least one hand is detected
        hand = hands[0]  # Get data for the first detected hand
        x, y, w, h = hand['bbox']  # Extract bounding box coordinates (x, y, width, height)

        # Create a blank white image of size 300x300
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region from the frame with added padding
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape  # Get the shape of the cropped image

        # Calculate aspect ratio of the hand region
        aspectRatio = h / w

        # If height is greater than width (tall aspect ratio)
        if aspectRatio > 1:
            k = imgSize / h  # Scaling factor based on height
            wCal = math.ceil(k * w)  # Calculate the new width after scaling
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize the image
            imgResizeShape = imgResize.shape  # Get the resized image shape
            wGap = math.ceil((imgSize - wCal) / 2)  # Calculate horizontal gap for centering
            imgWhite[:, wGap: wCal + wGap] = imgResize  # Place the resized image on the white canvas

        # If width is greater than or equal to height (wide aspect ratio)
        else:
            k = imgSize / w  # Scaling factor based on width
            hCal = math.ceil(k * h)  # Calculate the new height after scaling
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize the image
            imgResizeShape = imgResize.shape  # Get the resized image shape
            hGap = math.ceil((imgSize - hCal) / 2)  # Calculate vertical gap for centering
            imgWhite[hGap: hCal + hGap, :] = imgResize  # Place the resized image on the white canvas

        # Display the cropped and processed images
        cv2.imshow('ImageCrop', imgCrop)  # Show the cropped hand region
        cv2.imshow('ImageWhite', imgWhite)  # Show the processed 300x300 image

    # Display the original frame
    cv2.imshow('Image', img)

    # Wait for key press and check if "s" is pressed to save the image
    key = cv2.waitKey(1)
    if key == ord("s"):  # If "s" key is pressed
        counter += 1  # Increment the image counter
        # Save the processed image with a unique timestamp-based filename
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)  # Print the counter value to the console