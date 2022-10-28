import tkinter as tk 
import customtkinter as ck 
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk 


window = tk.Tk()
window.geometry("720x680")
window.title("GYM assistant by VP_TM") 
ck.set_appearance_mode("dark")
img = tk.Frame(height=480, width=720)
img.place(x=10, y=90) 
lmain = tk.Label(img) 
lmain.place(x=0, y=0) 
 
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
        #here you can add something to happen when count hit's 0 like an alerting sound
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

buttonRep10 = ck.CTkButton(window, text='Rep = 10', command=open, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRep10.place(x=150, y=600)
buttonRep15 = ck.CTkButton(window, text='Rep = 15', command=test, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRep15.place(x=340, y=600)
buttonRep20 = ck.CTkButton(window, text='Rep = 15', command=test, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="red")
buttonRep20.place(x=530, y=600)

buttonrep = ck.CTkButton(window, text='RESET', command=test, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
buttonrep.place(x=10, y=600)

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



cap = cv2.VideoCapture(0,apiPreference=cv2.CAP_MSMF)
while True:
    success, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    img = img[:, :720, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    if cv2.waitKey(10) & 0xFF == ord('q'):
           break   
    
    window.update()








