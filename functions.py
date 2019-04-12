import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import distance


import math

def retimg(x,y,h,w,img):
    img1=img[max(0,int(y-h/2)):min(img.shape[0],int(y+h/2)),max(0,int(x-w/2)):min(int(x+w/2),img.shape[1])]
    return img1


def list_expansion_(Y,X_size):
    expanded_list = list(Y)
    for i in range(X_size):
        expanded_list.append(0)
    return expanded_list

def convl(lbp):
    lbp1=[]
    for l in lbp:
        s=0
        for i in l:
            s+=i
    lbp1.append(s)
    #print(len(lbp1))
    return lbp1
    
def convh(hef):
    href1=[]
    p=0
    sum=0
    for h in range(100):
        elem=0
        while (p<1980 and elem<21):
            sum+=hef[p]
            p+=1
            elem+=1
        href1.append(sum)
    #print(len(href1))
    return href1

def norm(x):
    sum=np.sum(x)/len(x)
    x=x/sum
    return x

e=math.e
pi=math.pi

def Sigmoid(arr , sig):
    xp=[]
    for i in arr:
        val=(e**((-1*(i**2))/(2*(sig**2))))/(math.sqrt(2*pi)*sig)
        xp.append(val)
    return xp

