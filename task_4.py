#I used the color converting example and bounding box example from the instructions.
#I added a for loop to go through all the produced contours
#I also added a minimum contour area to not draw hundreds of contours
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    thresh_lo = np.array([0,0,0])
    thresh_hi = np.array([179,255,60])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, thresh_lo, thresh_hi)

    # Find contours of masked image
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 5000
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        if (w*h > min_area):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("image", frame)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()