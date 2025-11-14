import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    if not ret:
        break


    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #Red Mask
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])
    mask_red1 = cv2.inRange(hsv,lower_red1,upper_red1)


    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])
    mask_red2 = cv2.inRange(hsv,lower_red2,upper_red2)

    mask_red = mask_red1 + mask_red2

    #Blue Mask
    lower_blue = np.array([94,80,2])
    upper_blue = np.array([126,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


    # Clean Masks
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)


    #Find red contours
    cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Draw red contours
    for c in cnts_red:
        area = cv2.contourArea(c)
        if area > 500:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, "RED DETECTED", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255),2)
    
     #Draw blue contours
    for c in cnts_blue:
        area = cv2.contourArea(c)
        if area > 500:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(frame, "BLUE DETECTED", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0, 0),2)
        
    
    cv2.imshow("FRAME", frame)
    cv2.imshow("RED MASK", mask_red)
    cv2.imshow("BLUE MASK", mask_blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()