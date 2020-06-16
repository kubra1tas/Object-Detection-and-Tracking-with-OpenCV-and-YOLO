#   KUBRA TAS 06 / 2020
#   OBJECT DETECTION AND TRACKING WITH OPENCV AND YOLO
#   FOR ASISGUARD AS



import numpy as np
from imutils.video import VideoStream
import time
import cv2
import os
from random import randint


# OpenCV version  check
version = cv2.__version__.split('.')[0]
print("OpenCV Version : ", version)

# Connect to laptop screen
cap = VideoStream(src=0).start()
time.sleep(1.0)

# Defining a flag and reference point, to be used to check mouse clicks
firstFlag = False
refPt = (0, 0)


# Fct to be used for tracking mouse left button click, to track one chosen item
def click_track(event, x, y, flags, prm):
    # global prm.s for tracking the mouse event flag and mouse click position
    global refPt, firstFlag

    # If the event of the mouse is the left click, convert the flag to high
    # and send the position info of the mouse
    if event == cv2.EVENT_FLAG_LBUTTON:
        firstFlag = True
        refPt = (x, y)


# Fct to be used for going back to the original state of the tracking
def click_release(event, x, y, flags, prm):
    # global prm.s for tracking the mouse event flag and mouse click position
    global refPt, firstFlag

    # If the event of the mouse is the second left click, convert the flag to low
    # and send the position info of the mouse
    if event == cv2.EVENT_FLAG_LBUTTON:
        firstFlag = False
        refPt = (x, y)


# Define Background Subtractor for pattern matching
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define a var named pos and txt file, for recording the position info
# of the items detected
pos = []
filewrite = open("coordinates.txt", 'w')

print('Hi!, What type of the method would you like to use?', '\n')
print('Type DL for Deep Learning (YOLO)', '\n')
print('Type PM for Pattern Matching (OpenCV-only)', '\n')
choice = input('Please select your tracker: ')
if choice == 'DL':
    dl = True
    patternMatch = False
elif choice == 'PM':
    dl = False
    patternMatch = True
else:
    print('Please enter one of the options that is given !')
    quit()

if dl:
    while (cap != False):
        # Check if hte mouse left is clicked,
        # if yes, call click_track function
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", click_track)

        # Continuous image acquisition
        frame = cap.read()

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                   dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
        configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

        # load YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # extract the image shape
        (H, W) = frame.shape[:2]

        # determine only the *output* layer names that is needed from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving the bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize the lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the
                    # size of the frame, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update the list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        # confidence-threshold
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                pos = str((x, y, w, h))
                filewrite.write(("Detected " + LABELS[classIDs[i]] + " object coordinate:"))
                filewrite.write(pos)
                filewrite.write("\n")
        else:
            # in case of no item is found in the frame
            text = "No Item is Found"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color, 2)

            # show the output frame
            cv2.imshow("frame", frame)
            cv2.waitKey(10)

        # check when the mouse clicked for an object 
        if firstFlag:

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.namedWindow("frame")
            cv2.setMouseCallback("frame", click_release)
            
            # get tge position info of the mouse 
            mX = refPt[0]
            mY = refPt[1]

            if len(idxs) > 0:
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # check if the mouse click is inside of the box
                    if x < mX & mX < (x + w):
                        if y < mY & mY < (y + h):

                            label2 = LABELS[classIDs[i]]
                            
                            # track the object as long as the mouse is clicked second time
                            while firstFlag:
                                pos = str((x, y, w, h))
                                filewrite.write("Detected coordinate:")
                                filewrite.write(pos)
                                filewrite.write("\n")

                                frame = cap.read()
                                frame = frame
                                (H, W) = frame.shape[:2]

                                # determine only the *output* layer names that we need from YOLO
                                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                                             swapRB=True, crop=False)
                                net.setInput(blob)
                                layerOutputs = net.forward(ln)


                                # initialize our lists of detected bounding boxes, confidences, and
                                # class IDs, respectively
                                boxes = []
                                confidences = []
                                classIDs = []

                                # loop over each of the layer outputs
                                for output in layerOutputs:
                                    # loop over each of the detections
                                    for detection in output:
                                        # extract the class ID and confidence (i.e., probability) of
                                        # the current object detection
                                        scores = detection[5:]
                                        classID = np.argmax(scores)
                                        confidence = scores[classID]

                                        # filter out weak predictions by ensuring the detected
                                        # probability is greater than the minimum probability
                                        if confidence > 0.5:
                                            # scale the bounding box coordinates back relative to the
                                            # size of the frame, keeping in mind that YOLO actually
                                            # returns the center (x, y)-coordinates of the bounding
                                            # box followed by the boxes' width and height
                                            box = detection[0:4] * np.array([W, H, W, H])
                                            (centerX, centerY, width, height) = box.astype("int")

                                            # use the center (x, y)-coordinates to derive the top and
                                            # and left corner of the bounding box
                                            x = int(centerX - (width / 2))
                                            y = int(centerY - (height / 2))

                                            # update our list of bounding box coordinates, confidences,
                                            # and class IDs
                                            boxes.append([x, y, int(width), int(height)])
                                            confidences.append(float(confidence))
                                            classIDs.append(classID)

                                # apply non-maxima suppression to suppress weak, overlapping bounding
                                # boxes
                                # confidence-threshold
                                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                                                        0.3)
                                if len(idxs) > 0:
                                    for j in idxs.flatten():
                                        # extract the bounding box coordinates
                                        (x, y) = (boxes[j][0], boxes[j][1])
                                        (w, h) = (boxes[j][2], boxes[j][3])

                                        label = LABELS[classIDs[j]]

                                        if label == label2:
                                            # draw a bounding box rectangle and label on the frame
                                            color = [int(c) for c in COLORS[classIDs[j]]]
                                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                            text = "{}: {:.4f}".format(LABELS[classIDs[j]], confidences[j])
                                            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.5, color, 2)
                                            # show the output frame
                                            cv2.imshow("frame", frame)
                                            cv2.waitKey(10)

                                            pos = str((x, y, w, h))
                                            filewrite.write(("Detected " + label + " object coordinate:"))
                                            filewrite.write(pos)
                                            filewrite.write("\n")


                                        else:
                                            text = "Target is out of frame bound"
                                            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                                        1.5, color, 2)
                                            # show the output frame
                                            cv2.imshow("frame", frame)
                                            cv2.waitKey(10)
                                else:
                                    text = "No Item is Found"
                                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                                1.5, color, 2)
                                    # show the output frame
                                    cv2.imshow("frame", frame)
                                    cv2.waitKey(10)
                        else:
                            firstFlag = False
                    else:
                        firstFlag = False
        # quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # show the output frame

        cv2.imshow("frame", frame)
        cv2.waitKey(10)



elif patternMatch:

    while (cap != False):
        # Check if hte mouse left is clicked,
        # if yes, call click_track function
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", click_track)

        # Continuous image acquisition
        frame = cap.read()

        # apply background substraction
        fgmask = fgbg.apply(frame)

        (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # looping for contours
        if contours != None :
            for c in contours:
                if cv2.contourArea(c) < 15000 :
                    continue

                else:
                    # get bounding box from countour
                    (x, y, w, h) = cv2.boundingRect(c)
                    # draw bounding box
                    rct = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    y = y - 15 if y - 15 > 15 else (y + 15)
                    text = (str(x) + "," + str(y) + "," + str(w) + "," + str(h))
                    cv2.putText(frame, text, (x, y), fontFace=3, fontScale=1,
                                color=(0, 0, 255), thickness=3)
                    
                    # check when the mouse clicked for an object 
                    if firstFlag:
                        
                        # get tge position info of the mouse 
                        mX = refPt[0]
                        mY = refPt[1]

                        # check if the mouse click is inside of the box
                        if x < mX & mX < (x + w):
                            if y < mY & mY < (y + h):

                                frame = cap.read()
                                rects = []
                                colors = []
                                # define a tracker
                                tracker = cv2.TrackerMOSSE_create()

                                # draw rectangles, select ROI, open new window
                                rect_box = (x, y, w, h)
                                rects.append(rect_box)
                                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

                                multitracker = cv2.MultiTracker_create()

                                multitracker.add(tracker,
                                                 frame,
                                                 rect_box)

                                # track the object as long as the mouse is clicked second time
                                while (firstFlag):
                                    frame = cap.read()

                                    # update location objects
                                    success, boxes = multitracker.update(frame)

                                    # draw the object tracked
                                    for i, newbox in enumerate(boxes):
                                        pts1 = (int(newbox[0]),
                                                int(newbox[1]))
                                        pts2 = (int(newbox[0] + newbox[2]),
                                                int(newbox[1] + newbox[3]))
                                        cv2.rectangle(frame, pts1, pts2, colors[i], 2, 1)

                                    # Close the frame
                                    if cv2.waitKey(10) & 0xFF == ord("q"):
                                        break

                                    # display frame
                                    cv2.imshow("frame", frame)
                                    cv2.waitKey(10)

                                    cv2.namedWindow("frame")
                                    cv2.setMouseCallback("frame", click_release)

                                    mX = refPt[0]
                                    mY = refPt[1]
                                    pos = str((newbox[0], newbox[1], newbox[2], newbox[3]))
                                    filewrite.write("Detected specific object coordinate:")
                                    filewrite.write(pos)
                                    filewrite.write("\n")

                                    #Check if the second mouse click is located anywhere
                                    #but not on one of the objects
                                    (x, y, w, h) = cv2.boundingRect(c)
                                    if (x * 0.1) < mX & mX < (x + w * 1.1):
                                        if (y * 0.1) < mY & mY < (y + h * 1.1):
                                            firstFlag = True

                                        else:
                                            firstFlag = False
                                    else:
                                        firstFlag = False

                                    # Close the frame
                                    if cv2.waitKey(1) & 0xFF == ord("q"):
                                        break

                        else:
                            firstFlag = False
                    else:
                        firstFlag = False

                    cv2.imshow("frame", frame)
                    pos = str((x, y, w, h))
                    filewrite.write("Detected coordinate:")
                    filewrite.write(pos)
                    filewrite.write("\n")
        else:
            # in case of no item is found in the frame
            text = "No Item is Found"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color = (0, 0, 255), thickness=3)

            # show the output frame
            cv2.imshow("frame", frame)
            cv2.waitKey(10)



        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



else:
    pass


filewrite.close()
cv2.destroyAllWindows()



