import numpy as np
import cv2
import time
import mediapipe as mp

# Globals
DIM = 250
draw_color = (255, 255, 255)  # Color for drawing
erase_color = (0, 0, 0)        
prev_x, prev_y = 0, 0

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
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist


# Draws the axis for vector graphing    
def axis(img):
    thickness = 3
    cv2.line(img, (img.shape[1]-DIM, 0), (img.shape[1]-DIM, DIM), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-DIM, DIM), (img.shape[1], DIM), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), 0), (img.shape[1]-(DIM*2), DIM), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), DIM), (img.shape[1]-DIM, DIM), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), DIM), (img.shape[1]-(DIM*2), (DIM*2)), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-(DIM*2), (DIM*2)), (img.shape[1]-DIM, (DIM*2)), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-DIM, DIM), (img.shape[1]-DIM, (DIM*2)), (0, 0, 255), thickness)
    cv2.line(img, (img.shape[1]-DIM, (DIM*2)), (img.shape[1], (DIM*2)), (0, 0, 255), thickness)
    return img

def draw_line(cap,start,end,color, thickness=2):
    cv2.line(cap, start, end, color, -1)


def erase_area(cap, center, radius, color):
    cv2.circle(cap, center, radius, color, -1)


def main():


    pTime = 0
    cTime = 0



      
    
    # this initalizes what camera to use if an external camera is not used it switches to laptop cam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    
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
        if len(lmlist) >= 8:
            print(lmlist[8])
        
        # calculate the frames
        cTime = time.time()
        if cTime - pTime > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(frame_rgb)

        # Draw landmarks and get hand positions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    # Get x, y coordinates of each landmark
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 8:  # Index finger tip (Left hand)
                        # Use index finger to draw
                        if prev_x != 0 and prev_y != 0:
                            draw_line(cap, (prev_x, prev_y), (cx, cy), draw_color)
                        prev_x, prev_y = cx, cy

                    elif id == 12:  # Index finger tip (Right hand)
                        # Use middle finger to erase
                        erase_area(cap, (cx, cy), 20, erase_color)

        
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
