import numpy as np
import cv2
import time
import mediapipe as mp
import pyautogui

# Globals
DIM = 250


# Hand tracking class to be able to know at what locations every point is at
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    # Puts the hand markers on the frame
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # returns a 2D list with the locations with x y coordinates of every point
    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    #cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                    cv2.putText(img,str(int(id)),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)
        return lmlist


# Draws the axis for vector graphing    
def axis(img):
    thickness = 3
    color = (0, 0, 255)
    cv2.line(img, (img.shape[1]-DIM, 0), (img.shape[1]-DIM, DIM), color, thickness)
    cv2.line(img, (img.shape[1]-DIM, DIM), (img.shape[1], DIM), color, thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), 0), (img.shape[1]-(DIM*2), DIM), color, thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), DIM), (img.shape[1]-DIM, DIM), color, thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), DIM), (img.shape[1]-(DIM*2), (DIM*2)), color, thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), (DIM*2)), (img.shape[1]-DIM, (DIM*2)), color, thickness)
    cv2.line(img, (img.shape[1]-DIM, DIM), (img.shape[1]-DIM, (DIM*2)), color, thickness)
    cv2.line(img, (img.shape[1]-DIM, (DIM*2)), (img.shape[1], (DIM*2)), color, thickness)
    return img

def cursor(img,points):
    thickness = 1
    offset = 15
    color = (0, 255, 0)
    cv2.line(img, (points[1]-offset, points[2]), (points[1]+offset,points[2]), color, thickness)
    cv2.line(img, (points[1], points[2]-offset), (points[1], points[2]+offset), color, thickness)
    return img

def main():
    pTime = 0
    cTime = 0
    #Timers for the timed functions
    clickTimer = 0
    keyTimer = 0
    
    # this initalizes what camera to use if an external camera is not used it switches to laptop cam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_Width, screen_Height = pyautogui.size()
    xScale = screen_Width/frame_width
    yScale = screen_Height/frame_height

    
    # hand detection class
    detector = handDetector()
    
    # the first vector is the origin vector 
    # vectors will be formed from the orgin to any place in the graph
    # vectors is x distance from origin, y distance from origin, angle made
    success, img = cap.read()
    vecs = np.array([img.shape[1]-DIM,DIM,0], dtype=float)
    while True:
        #get
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = axis(img)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        
        #Gesture detection block 
        #thumb, index, middle, ring, and pinkie are true if they are extended and false if they aren't
        thumb = False
        index = False
        middle = False
        ring = False
        pinkie = False

        if len(lmlist) >= 8:
            if lmlist[4][1]<lmlist[3][1]:
                thumb = True
            if lmlist[8][2]<lmlist[5][2]:
                index = True
            if lmlist[12][2]<lmlist[9][2]:
                middle = True
            if lmlist[16][2]<lmlist[13][2]:
                ring = True
            if lmlist[20][2]<lmlist[17][2]:
                pinkie = True

        #If the thumb and index finger are extended, it moves the cursor to the fingertip
        if thumb and index:
            cursor(img,lmlist[8])
            pyautogui.moveTo(lmlist[8][1]*xScale, lmlist[8][2]*yScale)
            #if the middle finger is also extended for 3 seconds, clicks where the cursor is
            if middle:
                if clickTimer == 0:
                    clickTimer=time.time()
                elif time.time()-clickTimer >=3:
                    pyautogui.click()
                    clickTimer = 0 
            else :
                clickTimer=0
                
        #if the index, middle, and ring are up for 3 seconds then toggles the windows onscreen keyboard
        if index and middle and ring:
            if keyTimer == 0:
                keyTimer=time.time()
            elif time.time()-keyTimer >=3:
                pyautogui.hotkey('ctrl','win', 'o')
                keyTimer = 0 
        else :
            keyTimer = 0
            
        # stops the program if the index and pinkie are out and no other fingers are up
        if (index and pinkie) and not (thumb or middle or ring):
            break
                

        #if len(lmlist) >= 8:
        #    print(lmlist[8])
        
        # calculate the frames
        cTime = time.time()
        if cTime - pTime > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        
        # The settings text and fps
        cv2.rectangle(img, (5,5), (220, 65), (255, 0, 255) , 3 )
        cv2.putText(img, "Settings", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(img, str(int(fps)),(10, img.shape[0] - 30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0, 255),3)
        cv2.imshow("Image", img)

        # When you press q it quits the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
