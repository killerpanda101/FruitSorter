import cv2
import numpy as np


#capture the video

cap=cv2.VideoCapture(0)

while(1):
    ret,img=cap.read()

    
    #convert input image to BGR to HSV
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


    #Define the range of red color(for pomogarnate)
    red_lower=np.array([136,87,111],np.uint8)
    red_upper=np.array([180,255,255],np.uint8)

    #Define the range of purple color(for grape)
    Purple_lower=np.array([99,115,150],np.uint8)
    Purple_upper=np.array([110,255,255],np.uint8)

    #Define the range of yellow color(for banana)
    yellow_lower=np.array([22,60,200],np.uint8)
    yellow_upper=np.array([50,255,255],np.uint8)

    #Finding the range of RED PURPLE YELOW in image
    red=cv2.inRange(hsv,red_lower,red_upper)
    Purple=cv2.inRange(hsv,Purple_lower,Purple_upper)
    yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)

    #Morphological transformation , Dilation
    kernal=np.ones((5,5),"uint8")

    red=cv2.dilate(red,kernal)
    res=cv2.bitwise_and(img,img,mask=red)
    
    Purple=cv2.dilate(Purple,kernal)
    res1=cv2.bitwise_and(img,img,mask=Purple)

    yellow=cv2.dilate(yellow,kernal)
    res2=cv2.bitwise_and(img,img,mask=yellow)

    

     # Tracking the red color
    (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area>300):
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,"Pomegranate",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

     # Tracking the Purple color
    (_,contours,hierarchy)=cv2.findContours(Purple,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area>300):
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"Grapes",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0))
     # Tracking the yellow color
    (_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area>300):
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Banana",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))


    

    
    
    #cv2.imshow("red",Purple)
    #cv2.imshow("red",res)
    cv2.imshow("Color Tracking",img)
    

    

    

    key=cv2.waitKey(200)
    if key==32:
        break

cap.release()
cv2.destroyAllWindows()

    
    
