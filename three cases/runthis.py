import matplotlib.pyplot as plt
import cv2
from functions import Sigmoid as sigmoid , norm as norm
from mainfunc import reference as reference, finarr as finarr
import pdb
pathArr=[[63,78,62,25,'C:/Users/arshika/Desktop/mini project/Jogging0001.jpg'],[120,151,111,30,'C:/Users/arshika/Downloads/transferred content/arsh/jogging/images/Jogging0078.jpg'],[137,142,87,30,'C:/Users/arshika/Downloads/transferred content/arsh/jogging/images/Jogging0074.jpg']]

rArr=reference()

uneccess=[]
for p in pathArr:
    i=1
    x=p[0]
    y=p[1]
    h=p[2]
    w=p[3]
    path=p[4]
    plotArr=finarr(x,y,h,w,path,rArr)
    print("half")
    for j in range(1):
        
        shift=plotArr[j][0]
        l=plotArr[j][1]
        if j==0:
            ind=l.index(max(l))
            shif=shift[ind]
        elif j==1:
            ind1=l.index(max(l))
            shif1=shift[ind]
        else:
            ind2=l.index(max(l))
            shif2=shift[ind]
        sig=0.2
        l=norm(l)
        
        
        plt.plot(shift,sigmoid(l,sig),label=j)
        if j==0:
            print("the values of mixed vector l is")
            print(sigmoid(l,sig))
            img=cv2.imread(path)
        
        
        
    plt.legend(loc="best")
    
    plt.show()
    
    
    xn=x+shif
    xn1=x+shif1
    xn2=x+shif2
    yn=y
    hn=h
    wn=w
    
    cv2.rectangle(img,(xn1,yn),(xn1+wn,yn+hn),(255,255,0),2)
    cv2.rectangle(img,(xn2,yn),(xn2+wn,yn+hn),(0,0,255),2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img,(xn,yn),(xn+wn,yn+hn),(0,255,0),2)
    
    cv2.imshow("final",img)
    cv2.waitKey(25)

    
    
