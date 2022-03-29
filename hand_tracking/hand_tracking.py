import cv2
import mediapipe as mp
import time

# camera
cap = cv2.VideoCapture(0) #? open a video file or camera

# hands
mpHands = mp.solutions.hands # abbreviation for easy use
hands = mpHands.Hands() # create hands object
                        #! parameters for modifications
                        #* static_image_mode=False, track and detect base on confidence level (default)
                        #* static_image_mode=True, will do detection part every time, will be more slow
                        #? max_num_hands=2,
                        #? model_complexity=1,
                        #* min_detection_confidence=0.5, will try to detect if there's no tracking
                        #* min_tracking_confidence=0.5): will continue tracking until confidence go below 0.5

mpDraw = mp.solutions.drawing_utils #? function to draw points and lines in the hands landmarks

pTime = 0 # Previous time
cTime = 0 # Current time

# checking if camera has opened
assert cap.isOpened(), "file/camera could not be opened!"
while True:
    # capturing image
    success, img = cap.read()    #? returns a boolean and image content, gives a frame in BGR
    if not success: break        # if the camera status is false or null
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converts BGR to RGB

    #* sending the image to the process method of mpHands object (line 10)
    results = hands.process(imgRGB) #? results = <class 'mediapipe.python.solution_base.SolutionOutputs'>
                                    #? the hands object only receives RGB images

    # extracting the information from the results of process method
    if results.multi_hand_landmarks: #* if a hand it's detected
        for handlms in results.multi_hand_landmarks: # for each hand detected
            for id, lm in enumerate(handlms.landmark): # get the id and coordinates of each hand
                # print(id,lm) # the coordinates are given as ratio of the image
                height, width, channels = img.shape #* getting the pixel value of the image
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(id, cx, cy)

                # using the information
                if id == 8:
                    # drawing a circle(image, position, radius, color, thickness)
                    cv2.circle(img,(cx,cy),8,(255,0,255),cv2.FILLED)

            #? drawing the hands in the image
            #? draw_landmarks(image(where to draw), landmarks(hands), hand_connections(optional))
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)


    cTime = time.time() #in every iteration there is delta of time
    fps = 1/(cTime-pTime) # calculating how many time per second the iteration it's run
    pTime = cTime

    #*  displaying the image imshow(window_name, image)
    #? writing text in the image, putText(image, text, (x,y)position, Font, scale, color)

    cv2.putText(img, str(int(fps)), (15,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("image", img)             # standard procedure to run a webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):# standard procedure to run a webcam
      break