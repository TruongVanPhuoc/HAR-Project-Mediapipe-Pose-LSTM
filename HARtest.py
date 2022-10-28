
from ast import Str
from operator import imod
from tkinter import font

import cv2
from matplotlib.pyplot import text
import mediapipe as mp
from numpy import around
import pandas as pd
import datetime
# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Khởi tạo thư viện mediapipe
while cap.isOpened():
    ret, img = cap.read()
    timess= datetime.datetime.now()
    timenow= str(timess.hour)+":"+str(timess.minute)+":"+str(around(timess.second))
    daynow = str(timess.day)+"/"+str(timess.month)+"/"+str(timess.year)         
    cv2.rectangle(img, (0,0), (900,50), (245,117,16), -1)
    cv2.rectangle(img, (0,0), (120,150), (245,117,16), -1)
    cv2.putText(img, daynow,(535,12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(img, timenow,(535,28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)    
    cv2.putText(img,'Action',(300,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)          
        # Rep data
    cv2.putText(img, 'COUNTER', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        
        # Stage data
    cv2.putText(img, 'STAGE', (100,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow('Mediapipe Feed', img)                
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()