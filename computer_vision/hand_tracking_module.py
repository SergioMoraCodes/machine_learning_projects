import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False, max_hands=2, detectconf=0-5, trackconf=0.5):
        self.mode                     = mode
        self.max_hands                = max_hands
        self.min_detection_confidence = detectconf
        self.min_tracking_confidence  = trackconf

        self.mpHands = mp.solutions.hands
        self.hands   = self.mpHands.Hands(static_image_mode= self.mode,
                                          max_num_hands = self.max_hands,
                                          min_detection_confidence= self.min_detection_confidence,
                                          min_tracking_confidence= self.min_tracking_confidence)
        self.mpDraw  = mp.solutions.drawing_utils
        # self.results = None

    def findhands(self,img, draw= True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks and draw:
            for handlms in self.results.multi_hand_landmarks:

                self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handnum=0, draw= True):

        lmlist = []

        if self.results.multi_hand_landmarks and draw:
            myhand = self.results.multi_hand_landmarks[handnum]
            for id, lm in enumerate(myhand.landmark):
                height, width, channels = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                lmlist.append([id, cx, cy])

        return lmlist

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success: break
        img = detector.findhands(img)
        # lmlist = detector.findposition(img)
        # if len(lmlist) != 0:
        #     print(lmlist[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (15,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()