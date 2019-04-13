import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import feature
import functions
import math

from functions import Sigmoid as sigmoid,norm as norm, retimg as retimg , convl as convl, convh as convh
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
print(lbp)
print(len(lbp))
href = feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
href=convh(href)
print(len(href))
print(href)
#full occlusion
fullocl=cv2.imread('C:/Users/arshika/Downloads/transferred content/arsh/jogging/images/Jogging0074.jpg')

D = []
shift = []

x=137
y=142
w=30
h=87

x_init = x

y_init = y 


for i in range(10):
    
    x=x+10
    shift.append(x-x_init)
    comp=retimg(x,y,h,w,fullocl)
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

print(D)
print(shift)


x_init = x

y_init = y 


D1 = []
shift1 = []

#at shift=0
shift1.append(x-x_init)
comp=retimg(x,y,h,w,fullocl)
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


for i in range(10):
    
    x=x-10
    shift1.append(x-x_init)
    comp=retimg(x,y,h,w,fullocl)
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

D1.reverse()
shift1.reverse()
print(D1)
print(shift1)

D=D1+D
shift=shift1+shift

print(D)
print(shift)

plt.plot(shift,D)
plt.show()

D=norm(D)
plt.plot(shift,D)  
plt.show()

for s in [ 0.2]:
    plt.plot(shift, sigmoid(D,s))
plt.show()



