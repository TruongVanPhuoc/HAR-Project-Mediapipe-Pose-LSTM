
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from cvzone.PoseModule import PoseDetector
import pyglet
import datetime

detector = PoseDetector()

label = "Warmup...."
n_time_steps = 10
lm_list = []
comHand= pyglet.media.load('comHand.wav')
comPull= pyglet.media.load('comPullup.wav')
comPush= pyglet.media.load('comPushup.wav')
comsquat= pyglet.media.load('comSquat.wav')
mp_drawing = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
#model = tf.keras.models.load_model("modeltestsound.h5")
model = tf.keras.models.load_model("modeltestsound.h5")
print("Loaded model from disk")
cap = cv2.VideoCapture(0,apiPreference=cv2.CAP_MSMF)

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
stage = None
Rep=5 
while True:
    success, img = cap.read()
    # img = detector.findPose(img,draw=False)
    # imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
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
                        if counter >= Rep:
                          comPush.play()

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
  
                    # for i in counter2:
                    if counter2 >= Rep:
                          comHand.play()
                        
                    


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

                    if counter3 >= Rep:
                          comsquat.play()    

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
                    
                    if counter4 >= Rep:
                          comPull.play()    
                     
                     

        
    except:
        pass

        # Render curl counter
        # Setup status box
    timess= datetime.datetime.now()
    timenow= str(timess.hour)+":"+str(timess.minute)+":"+str( round(timess.second))
    daynow = str(timess.day)+"/"+str(timess.month)+"/"+str(timess.year) 
    cv2.rectangle(img, (0,0), (900,50), (245,117,16), -1)
    cv2.rectangle(img, (0,0), (120,150), (245,117,16), -1)
    cv2.putText(img, daynow,(535,12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, timenow,(535,28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)    
    
    ## List Counter
    if counter < Rep:
        cv2.putText(img,'Pushup:'+ str(counter) ,(5,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

    else:
         cv2.putText(img, 'Pushup:' + str(counter)+'Done!!!', (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    if counter2 < Rep:
        cv2.putText(img,'Hand:'+ str(counter2) ,(5,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'Hand:' + str(counter2)+'Done!!!', (5, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
      
   
    if counter3 < Rep:
        cv2.putText(img,'Squat:'+ str(counter3) ,(5,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'Squat:' + str(counter3)+'Done!!!', (5, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    if counter4 < Rep:
        cv2.putText(img,'Pullup:'+ str(counter4) ,(5,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'Pullup:' + str(counter4)+'Done!!!', (5, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    
    cv2.putText(img,'Action',(300,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, label, 
                    (300,40), 
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
    cv2.imshow('GYM assistant by VP_TM', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break    

cap.release()
cv2.destroyAllWindows()
