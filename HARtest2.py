import tkinter as tk 
import customtkinter as ck 
import numpy as np 
import cv2
import mediapipe as mp
import numpy as np
from cvzone.PoseModule import PoseDetector
import pyglet
import winsound as ws
import tkinter as tk
from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk 


window = tk.Tk()
window.title("GYM assistant by VP_TM") 
window.geometry("800x660")
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
    elif results[0][4] > 0.5:
         label ="Normal"
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


    img = img[:, :720, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    #lmain.after(10, detect)  
    window.update()




