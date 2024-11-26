import numpy as np
import cv2
import time
import mediapipe as mp
import pyautogui
from playsound import playsound

# Globals
DIM = 250
draw_color = (255, 255, 255)  # Color for drawing
erase_color = (0, 0, 0)        

# Hand tracking class to be able to know at what locations every point is at
class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
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
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # returns a 2D list with the locations with x y coordinates of every point
    def find_position(self, img, handNo=0, draw=True):
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

def draw_line(img, start, end, color, thickness=2):
    # Ensure the line is drawn within the defined axis
    x_min, y_min = img.shape[1] - DIM * 2, 0
    x_max, y_max = img.shape[1], DIM * 2
    
    # Clamp the start and end points to remain within the axis bounds
    start_x = max(x_min, min(start[0], x_max))
    start_y = max(y_min, min(start[1], y_max))
    end_x = max(x_min, min(end[0], x_max))
    end_y = max(y_min, min(end[1], y_max))
    
    # Draw the clamped line
    cv2.arrowedLine(img, (start_x, start_y), (end_x, end_y), color, thickness)
    return img

def draw_vectors(img, vecs):
    for vector in vecs:
        draw_line(img, [vecs[0][0], vecs[0][1]], [vector[0], vector[1]], (255,255,255))
    return img

def draw_popup(img, vectors):
    # Get the frame dimensions
    height, width, _ = img.shape

    # Popup dimensions
    popup_w, popup_h = 300, 200  # Width and height of the popup
    popup_x = width - popup_w  
    popup_y = height - popup_h  

    # Draw the popup background
    cv2.rectangle(img, (popup_x, popup_y), (popup_x + popup_w, popup_y + popup_h), (50, 50, 50), -1)
    cv2.rectangle(img, (popup_x, popup_y), (popup_x + popup_w, popup_y + popup_h), (255, 255, 255), 2)

    # Title
    cv2.putText(img, "Vector Info", (popup_x + 10, popup_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display vector details
    y_offset = 50
    for i, vector in enumerate(vectors):
        if i != 0:
            text = f"Vec {i}: x={vector[0] - vectors[0][0]}, y={-(vector[1] - vectors[0][1])}, angle={vector[2]:.2f}"
            cv2.putText(img, text, (popup_x + 10, popup_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20  # Spacing between lines

    return img

def draw_tutorial_popup(img, tutorial):
    popup_w, popup_h = 500, 200
    popup_x, popup_y = 20, 20

    if not tutorial:
        # Draw button for showing tutorial
        cv2.rectangle(img, (popup_x, popup_y), (popup_x + 185, popup_y + 50), (50, 50, 50), -1)
        cv2.rectangle(img, (popup_x, popup_y), (popup_x + 185, popup_y + 50), (255, 255, 255), 2)
        cv2.putText(img, "Show Tutorial", (popup_x + 10, popup_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # Draw tutorial popup
        cv2.rectangle(img, (popup_x, popup_y), (popup_x + popup_w, popup_y + popup_h), (50, 50, 50), -1)
        cv2.rectangle(img, (popup_x, popup_y), (popup_x + popup_w, popup_y + popup_h), (255, 255, 255), 2)

        # Tutorial content
        tutorial_text = [
            "Tutorial:",
            "Index & Thumb extended: Cursor",
            "   Add Middle (hold for 3 seconds): Selection",
            "Index & Pinkie (or hold Q key): Close app"
        ]

        # Display tutorial text line by line
        y_offset = popup_y + 30
        for line in tutorial_text:
            cv2.putText(img, line, (popup_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30  # Line spacing
    return img



def main():
    prev_x = 0
    prev_y = 0

    pTime = 0
    cTime = 0
    #Timers for the timed functions
    clickTimer = 0
    keyTimer = 0
    
    tutorial = False
    
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
    vecs = [[img.shape[1]-DIM,DIM,0]]
    show_popup = True
    while True:
        print("vecs:", vecs)

        #get
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = axis(img)
        img = detector.find_hands(img)
        lmlist = detector.find_position(img)
        img = draw_vectors(img, vecs)
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
            if lmlist[12][2]<lmlist[10][2]: # switch 9 with 10 so you dont have to bend down as far
                middle = True
            if lmlist[16][2]<lmlist[14][2]: # did the same with 13 switched with 14
                ring = True
            if lmlist[20][2]<lmlist[18][2]:
                pinkie = True

        # If the thumb and index finger are extended, it moves the cursor to the fingertip 
        # It will also put a line on the screen for a possibe vector
        if thumb and index:
            cursor(img,lmlist[8])
            # pyautogui.moveTo(lmlist[8][1]*xScale, lmlist[8][2]*yScale)
            pyautogui.moveTo(lmlist[8][1], lmlist[8][2])
            # if the middle finger is also extended for 3 seconds, clicks where the cursor is
            
            cx, cy = int(lmlist[8][1]), int(lmlist[8][2])
            inGraph = (cx >= (img.shape[1]-(2*DIM)) and cx <= img.shape[1]) and (cy >= 0 and cy <= (2*DIM))
            if inGraph:
                if prev_x != 0 and prev_y != 0:
                    draw_line(img, (img.shape[1]-DIM, DIM), (cx, cy), draw_color)
                prev_x, prev_y = cx, cy
            # if the middle finger is also extended for 3 seconds, clicks where the cursor is
            # it will place the vector down on screen
            if middle:
                if clickTimer == 0:
                    clickTimer=time.time()
                elif time.time()-clickTimer >=3:
                    pyautogui.click()
                    if len(vecs) < 3 and inGraph:
                        new_vector = [lmlist[8][1], lmlist[8][2], abs(np.degrees(np.arctan((-(lmlist[8][2]-vecs[0][1]))/(lmlist[8][1]-vecs[0][0]))))]
                        vecs.append(new_vector)
                    clickTimer = 0 
                    if not tutorial and (cx >= 20 and cx <= 205) and (cy >= 20 and cy <= 70):
                        tutorial = True
                    elif tutorial and (cx >= 20 and cx <= 520) and (cy >= 20 and cy <= 220):
                        tutorial = False
                    playsound('audios/ping-82822.mp3', False)
            else :
                clickTimer=0
        else: 
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
        
        # calculate the frames
        cTime = time.time()
        if cTime - pTime > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
                        
        if cv2.waitKey(10) & 0xFF == ord('p'):
            show_popup = not show_popup
            
        if show_popup:
            if len(vecs) > 1:
                img = draw_popup(img, vecs)

        # The Tutorial text and fps
        img = draw_tutorial_popup(img, tutorial)
        # if not tutorial: 
        #     cv2.rectangle(img, (5,5), (220, 65), (50, 50, 50) , -1 )
        #     cv2.rectangle(img, (5,5), (220, 65), (255, 255, 255) , 2 )
        #     cv2.putText(img, "Tutorial", (15, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        # else:
        #     s = "Tutorial\nIndex and Thumb extended: is for cursor\n\tAdd the Thumb: selection\nIndex and Pinkie (or q key): closes application"
        #     cv2.putText(img, s, (15, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
            
        
        cv2.putText(img, str(int(fps)),(10, img.shape[0] - 30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0, 255),3)
        cv2.imshow("Image", img)

        # When you press q it quits the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()


