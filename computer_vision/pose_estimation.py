import cv2             # image processing
import mediapipe as mp # framework for pose estimation
import time

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(
                    static_image_mode = False, # When is detecting or tracking, with is always detecting
                    smooth_landmarks  = True ,
                    min_detection_confidence = 0.5,
                    min_tracking_confidence  = 0.5 )
cap   = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read() # img in BGR
    if not success: break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(id, lm)
            cx, cy = int(lm.x*width), int(lm.y*height)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps   = 1/(cTime-pTime)
    pTime = cTime 

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow('pose estimation', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break