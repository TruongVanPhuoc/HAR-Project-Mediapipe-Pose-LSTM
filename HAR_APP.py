import tkinter as tk 
import customtkinter as ck 
import numpy as np 
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from cvzone.PoseModule import PoseDetector
import pyglet
import datetime
import winsound as ws
import tkinter as tk
from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk 
window = tk.Tk()
window.title("GYM assistant by VP_TM") 
window.geometry("800x600")
window.iconbitmap(default="LLogo.ico")


ck.set_appearance_mode("dark")
font , bg , fg  =("Century Gothic" , 15) , "#333" , '#fff'
def test():
       global Rep
       Rep == 5

def set_timer():
    global H , S , M
    H , M , S = get_seconds.get().split(':')

def countdown():
    Start_btn['stat'] = 'disabled'
    global S , M , H
    if  int(M) != 0 and int(H) != 0:
        H = int(H)
        M =int(M) 
        S = 59
        M = 59
        H -=1
        M -= 1
    if int(S) == 0 and int(H) == 0 and int(M) == 0 :
        Start_btn['stat'] = 'normal'
        count_lb['text'] = "00:00:00"
        H , S , M = 0,0,0 
        ws.PlaySound("endtime",ws.SND_FILENAME)
    elif int(S) == 0 :
        S = 59
        M = int(M)
        M -=1
        count_lb['text'] = "%s:%s:%s" % (H , int(M) , S ) 
        countdown()       
    else:
        timz = ( str(int(H)).zfill(2) , str(int(M)).zfill(2) , str(S).zfill(2))
        time_str = '%s:%s:%s' % timz 
        count_lb['text'] = time_str
        S = int(S) -1
        count_lb.after(1000,countdown)
def luanch():
    set_timer()
    countdown()

detector = PoseDetector()
label = "Warmup...."
n_time_steps = 10
lm_list = []
comHand= pyglet.media.load('comHand.wav')
comPull= pyglet.media.load('comPullup.wav')
comPush= pyglet.media.load('comPushup.wav')
comsquat= pyglet.media.load('comSquat.wav')
img = tk.Frame(height=480, width=720)
img.place(x=10, y=90) 
lmain = tk.Label(img) 
lmain.place(x=0, y=0) 
mp_drawing = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
#model = tf.keras.models.load_model("modeltestsound.h5")
model = tf.keras.models.load_model("model.h5")
print("Loaded model from disk")
cap = cv2.VideoCapture(0,apiPreference=cv2.CAP_MSMF)

def reset_counter(): 
    global counter
    global counter2
    global counter3
    global counter4
    counter = 0 
    counter2 = 0
    counter3 = 0
    counter4 = 0

def rep5():
    global Rep
    Rep=5
def rep10():
    global Rep
    Rep=10
def rep15():
    global Rep
    Rep=15
def repf():
    global Rep
    Rep=9999

buttonRep5 = ck.CTkButton(window, text='Rep =   5', command=rep5, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRep5.place(x=670, y=100)
buttonRep10 = ck.CTkButton(window, text='Rep = 10', command=rep10, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRep10.place(x=670, y=150)
buttonRep15 = ck.CTkButton(window, text='Rep = 15', command=rep15, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRep15.place(x=670, y=200)
buttonRepf = ck.CTkButton(window, text='Free  ' + '  ', command=repf, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRepf.place(x=670, y=250)

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=121, text_font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=670, y=300)



classLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=20 )
classLabel.place(x=35, y=35)
classLabel.configure(text='SET TIME') 

count_lb = Label(window,text = "00:00:00", fg=fg, bg = "#000" , font = (font[0] , 20))
count_lb.place(relx= 0.08+0.2 , rely = 0.03 , relwidth = 0.2, relheight = 0.1)
Start_btn= Button(window , text = "SET" ,font = font ,command = luanch , fg= fg, bg = bg ,relief ='flat')
Start_btn.place(relx= 0.4+0.2 , rely = 0.03 , relwidth = 0.2 , relheight = 0.1)
get_seconds = ttk.Entry(window ,font = font)
get_seconds.place(relx= 0.3+0.2 , rely = 0.03 , relwidth = 0.15 , relheight = 0.1)
get_seconds.insert(0,"00:00:00")


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (300, 70)
    fontScale = 2
    fontColor = (255, 0, 0)
    thickness = 3
    lineType = 5
    cv2.putText(img, label,bottomLeftCornerOfText,font,fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "Squat"
    elif results[0][1] > 0.5:
        label = "Hand"
    elif results[0][2] > 0.5:
        label = "Pushup"
    elif results[0][3] > 0.5:
        label = "Pullup"
    return label

i = 0
warmup_frames = 10
counter = 0 
counter2 = 0
counter3 = 0
counter4 = 0
calo = 0
calo2 = 0
calo3 = 0
calo4 = 0
caloth= 0
caloths= 0
stage = None
Rep=5 
while True:
    success, img = cap.read()
    # img = detector.findPose(img,draw=False)
    # imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(img)
    i = i + 1
    if i > warmup_frames:
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            #img = draw_landmark_on_image(mpDraw, results, img)
    

    #img = draw_class_on_image(label, img)
    try:
        
                landmarks = results.pose_landmarks.landmark
                 #Hand&Pushup
                shoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]
              
                
                shoulder2 = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow2 = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist2 = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
                
                #Squat
                hip = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x,landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].y]

                hip2 = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
                knee2 = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle2 = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                if label=="Pushup":
                        angle = calculate_angle(shoulder, elbow, wrist)

                        cv2.putText(img, str(round(angle,2)), 
                                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )
                        angle2 = calculate_angle(shoulder2, elbow2, wrist2)
                        cv2.putText(img, str(round(angle, 2)),
                                    tuple(np.multiply(elbow2, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        # Curl counter logic
                        if angle and angle2 > 100:
                                stage = "up"
                        if angle and angle2 < 90 and stage =='up':
                                stage="down"
                                counter +=1
                                calo = counter * 1.5

                
                        

                elif label=="Hand":

                    angle = calculate_angle(shoulder, elbow, wrist)

                    cv2.putText(img, str(round( angle ,2)), 
                                tuple(np.multiply(elbow , [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    angle2 = calculate_angle(shoulder2, elbow2, wrist2)
                    cv2.putText(img, str(round( angle2 ,2)), 
                                tuple(np.multiply(elbow2 , [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    
                    # Curl counter logic
                    if angle and angle2  > 140:
                        stage = "down"
                    if angle and angle2  < 50 and stage == "down":
                        stage="up"
                        counter2 +=1
                        calo2 = counter2 * 0.5

                    


                elif label=="Squat":
                    angle = calculate_angle(hip, knee, ankle)
                    cv2.putText(img, str(round( angle ,2)),
                                tuple(np.multiply(knee , [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    angle2 = calculate_angle(hip2, knee2, ankle2)
                    cv2.putText(img, str(round(angle2, 2)),
                                tuple(np.multiply(knee2, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Curl counter logic
                    if angle and angle2  > 170:
                        stage = "up"
                    if angle and angle2  < 95 and stage == "up":
                        stage="down"
                        counter3 +=1
                        calo3 = counter3 * 2

                elif label=="Pullup":
                    angle = calculate_angle(shoulder, elbow, wrist)
                    cv2.putText(img, str(round( angle ,2)),
                                tuple(np.multiply(elbow , [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    angle2 = calculate_angle(shoulder2, elbow2, wrist2)
                    cv2.putText(img, str(round( angle2 ,2)),
                                tuple(np.multiply(elbow2 , [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # Curl counter logic
                    if angle and angle2  > 160:
                        stage = "down"
                    if angle and angle2  < 60 and stage == "down":
                        stage="up"
                        counter4 +=1
                        calo4 = counter4 * 1.5
     

        
    except:
        pass

        # Render curl counter
        # Setup status box
    # caloth = calo + calo2 + calo3 + calo4
    # caloths= round(caloth/timess.second,2)
    timess= datetime.datetime.now()
    timenow= str(timess.hour)+":"+str(timess.minute)+":"+str( round(timess.second))
    daynow = str(timess.day)+"/"+str(timess.month)+"/"+str(timess.year) 
    cv2.rectangle(img, (0,0), (900,50), (245,117,16), -1)
    # cv2.rectangle(img, (0,0), (120,150), (245,117,16), -1)
    cv2.putText(img, daynow,(535,12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, timenow,(535,28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)    
    
    ## List Counter
    if counter < Rep:
        cv2.putText(img,'Pushup:'+ str(counter) ,(5,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2, cv2.LINE_AA)
    elif counter == Rep:
        ws.PlaySound("comPushup",ws.SND_FILENAME)
        counter = counter+1
        cv2.putText(img, 'Pushup:' + str(counter)+'Done!!!', (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)
    else:
         cv2.putText(img, 'Pushup:' + str(counter)+'Done!!!', (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)
    
    
    if counter2 < Rep:
        cv2.putText(img,'Hand:'+ str(counter2) ,(5,90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2, cv2.LINE_AA)
    elif counter2 == Rep:
        ws.PlaySound("comHand",ws.SND_FILENAME)
        counter2 = counter2+1
        cv2.putText(img, 'Pushup:' + str(counter)+'Done!!!', (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)              
    else:
        cv2.putText(img, 'Hand:' + str(counter2)+'Done!!!', (5, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)
    

    if counter3 < Rep:
        cv2.putText(img,'Squat:'+ str(counter3) ,(5,110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2, cv2.LINE_AA)
    elif counter3 == Rep:
        ws.PlaySound("comSquat",ws.SND_FILENAME)
        counter3 = counter3+1
        cv2.putText(img, 'Pushup:' + str(counter)+'Done!!!', (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)                
    else:
        cv2.putText(img, 'Squat:' + str(counter3)+'Done!!!', (5, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)
    
    
    if counter4 < Rep:
        cv2.putText(img,'Pullup:'+ str(counter4) ,(5,130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2, cv2.LINE_AA)
    elif counter4 == Rep:
        ws.PlaySound("comPushup",ws.SND_FILENAME)
        counter4 = counter4+1
        cv2.putText(img, 'Pushup:' + str(counter)+'Done!!!', (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, 'Pullup:' + str(counter4)+'Done!!!', (5, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,0,0), 2, cv2.LINE_AA)
    
    
    cv2.putText(img,'Action',(200,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, label, 
                    (200,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)     
        # Rep data
    cv2.putText(img, 'COUNTER', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    if label == "Pushup":
          cv2.putText(img, str(counter), 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif label == "Hand":
          cv2.putText(img, str(counter2), 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif label == "Squat":
          cv2.putText(img, str(counter3), 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif label == "Pullup":
          cv2.putText(img, str(counter4),
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    ###calo

    cv2.putText(img, 'Calories: ' + str(caloth) , (380, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Calo/s  :' + str(caloths), (380, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Stage data
    cv2.putText(img, 'STAGE', (100,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, stage, 
                    (100,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                 mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                  )        
    img = img[:, :720, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    #lmain.after(10, detect)  
    window.update()




