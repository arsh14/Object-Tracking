import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import feature
import functions
import math
from functions import Sigmoid as sigmoid,norm as norm, retimg as retimg , convl as convl, convh as convh


def reference():
    #ref img
    xo=63
    yo=78
    ho=62
    wo=25
    img=cv2.imread('C:/Users/arshika/Desktop/mini project/Jogging0001.jpg')
    comp=retimg(xo,yo,ho,wo,img)
    gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("imge", gray)
    frame = cv2.resize(gray, (50,100))
    lbp = feature.local_binary_pattern(frame , 8, 2, method="uniform")
    lbpref=convl(lbp)
    #print(lbp)
    #print(len(lbp))
    href = feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    href=convh(href)
    #print(len(href))
    #print(href)
    ret=[lbpref,href]
    return ret

def finarr(x,y,h,w,path,ret):
    lbpref,href=ret[0],ret[1]
    
    image=cv2.imread(path)
    
    #cv2.imshow('img',image)
    D = []
    Dl=[]
    Dh=[]
    shift = []
    x_init = x
    y_init = y

    for i in range(10):
        x=x+10
        shift.append(x-x_init)
        comp=retimg(x,y,h,w,image)
        gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(gray, (50,100))
        lbp = feature.local_binary_pattern(frame , 8, 2, method="uniform")
        lbp1=convl(lbp)
        h1 = feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        h1=convh(h1)
        Xnew=[x1 - x2 for (x1, x2) in zip(lbpref,lbp1)]
        Ynew=[x1 - x2 for (x1, x2) in zip(href,h1)]
        dist1=[]
        for a, b in zip(Xnew, Ynew):
            z=math.sqrt((a*a)+(b*b))
            dist1.append(z)
        L=0
        for i in dist1:
            L += i
        D.append(L)
        Dl.append(np.sum(Xnew))
        Dh.append(np.sum(Ynew))

    print(D)
    print(shift)
    x_init = x
    y_init = y 
    D1 = []
    shift1 = []
    Dl1=[]
    Dh1=[]

    #at shift=0
    shift1.append(x-x_init)
    comp=retimg(x,y,h,w,image)
    gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(gray, (50,100))
    lbp = feature.local_binary_pattern(frame , 8, 2, method="uniform")
    lbp1=convl(lbp)
    frame = cv2.resize(gray, (64,128))
    h1 = feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    h1=convh(h1)
    Xnew=[x1 - x2 for (x1, x2) in zip(lbpref,lbp1)]
    Ynew=[x1 - x2 for (x1, x2) in zip(href,h1)]
    dist1=[]
    for a, b in zip(Xnew, Ynew):
        z=math.sqrt((a*a)+(b*b))
        dist1.append(z)
    L=0
    for i in dist1:
        L += i
    D1.append(L)
    Dl1.append(np.sum(Xnew))
    Dh1.append(np.sum(Ynew))


    for i in range(10):
        
        x=x-10
        shift1.append(x-x_init)
        comp=retimg(x,y,h,w,image)
        gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(gray, (50,100))
        lbp = feature.local_binary_pattern(frame , 8, 2, method="uniform")
        lbp1=convl(lbp)
        frame = cv2.resize(gray, (64,128))
        h1 = feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        h1=convh(h1)
        Xnew=[x1 - x2 for (x1, x2) in zip(lbpref,lbp1)]
        Ynew=[x1 - x2 for (x1, x2) in zip(href,h1)]
        dist1=[]
        for a, b in zip(Xnew, Ynew):
            z=math.sqrt((a*a)+(b*b))
            dist1.append(z)
        L=0
        for i in dist1:
            L += i
        D1.append(L)
        Dl1.append(np.sum(Xnew))
        Dh1.append(np.sum(Ynew))

    D1.reverse()
    Dl1.reverse()
    Dh1.reverse()
    shift1.reverse()
    print(D1)
    print(shift1)

    D=D1+D
    Dl=Dl1+Dl
    Dh=Dh1+Dh
    shift=shift1+shift

    print(D)
    print(shift)

    #plt.plot(shift,D)
    #plt.show()
    plotArr=[[shift,D,x,y,h,w],[shift,Dl],[shift,Dh]]
    print("hey this is the shift")
    print(plotArr[0][0])
    print(plotArr[0][1])
    return plotArr     



