import numpy as np
import time
from functools import wraps
import cv2
import pyautogui
import collections
#import right

lastsave = 0

def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        global lastsave
        if time.time() - lastsave > 3:
            # this is in seconds, so 5 minutes = 300 seconds
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
right = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
left = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
#smile = cv2.CascadeClassifier('haarcascade_smile.xml')

@counter
def closed():
  print "Eye Closed"


def openeye():
  print "Eye is Open"

class slist(list):
    @property
    def length(self):
        return len(self)

count=0
counts=0
close=0
longPress=0
cam=cv2.VideoCapture(0)
imm = cv2.imread('E://download.jpg')

blank=np.zeros((480,848,3),dtype=np.uint8)  # Change this correctly to size of your image frame
fix=0

print "press y to set reference box for y motion" #set a reference initially for y motion

while(cam.isOpened()):


        ret,img = cam.read()
        r=0
        l=0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex,ey,ew,eh) in eyes:
                 print "Eyes Opened"
                 longPress=0 # while in eye open stage value will be 0
                 count +=1
                 print "======"
                 print count
                 print "======"
                 m=slist(range(count))
                 m.length
                 
            if(len(eyes)<=0 ):  #condition when eye are not find in the face.
                print "Eye closed"
                longPress = longPress+1 # if eye blink stage value will be increamneted by one in case eye in close stage at next iteration value will be greater then 1
                close +=1
                print "$$$$$$"
                print close
                print "$$$$$$"
                c=slist(range(close))
                c.length
            if(counts>=60):
                cv2.imshow("image", imm)
            if(longPress == 1): #if blink is long then call click event 
                pyautogui.click()
                print "Mouse clicked"
                counts+=1
                print counts
                s=slist(range(counts))
                s.length
                    
                    
   
        r_eye= right.detectMultiScale(gray, 1.9, 5)
        l_eye= left.detectMultiScale(gray, 1.9, 5)  #Change these values according to face distance from screen

        for (rx,ry,rw,rh) in r_eye:
                cv2.rectangle(img,(rx,ry),(rx+rw,ry+rh),(255,255,0),2)
                r_c=(rx+rw/2,ry+rh/2)
                r=1

        for (lx,ly,lw,lh) in l_eye:          
                cv2.rectangle(img,(lx,ly),(lx+lw,ly+lh),(0,255,255),2)
                l_c=(lx+lw/2,ly+lh/2)
                l=1

        if(r*l):

            if(l_c[0]-r_c[0]>50):
                cv2.line(img,r_c,l_c,(0,0,255),4)
                mid=((r_c[0]+l_c[0])/2,(r_c[1]+l_c[1])/2)
                cv2.circle(img,mid,2,(85,25,100),2)
                if(fix==1):                        # Change this part of code according to what you want
                                                   # for motion along y direction
                    if( mid[1]<one[1]):
                        pyautogui.moveRel(None, -15)
                    if(mid[1]>two[1]):
                        pyautogui.moveRel(None, 15)

                if(cv2.waitKey(1))== ord('y'):
                        blank=np.zeros_like(img)
                        one=(mid[0]-60,r_c[1]-7)   # Change the Value 60,7 to change box dimentions
                        two=(mid[0]+60,l_c[1]+7)   # Change the Value 60,7 to change box dimentions
                        cv2.rectangle(blank,one,two,(50,95,100),2)
                        fix=1


        elif(r) :   pyautogui.moveRel(-30, None)   # Change the Value and Sign to change speed and direction

        elif (l):   pyautogui.moveRel(30, None)    # Change the Value and Sign to change speed and direction

        if(cv2.waitKey(1))== ord('a'):
            print "total number of eye open",m
            print "total",len(m)
            print "total numberof closed eye",c
            print "total",len(c) 
            print "total number of clicks",s
            print "total",len(s)
            text_file=open("E:\\outputes.txt","w")
            text_file.write("total eyeopen: %d" % len(m))
            text_file.write("\ntotal eyeclose: %d" % len(c))
            text_file.write("\ntotal click: %d" % len(s))

        

        #img=cv2.bitwise_or(img,blank)
        rimg = cv2.flip(img,1)
        cv2.imshow('img',rimg)
        cv2.imshow('img1',blank)
        if(cv2.waitKey(1))==27:break
        

cv2.destroyAllWindows()
