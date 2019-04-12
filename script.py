import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as parser
from functions import Sigmoid as sigmoid , norm as norm
from mainfunc import reference as reference, finarr as finarr
import pdb

tree = parser.parse('C:/Users/arshika/Desktop/mini_project sm/images/grtruth.xml')
root = tree.getroot()

workDir="Jogging"
similarity=[]
startFrame=1
endFrame=306
objectNum=0
rArr=reference()
frameCnt=0
#pdb.set_trace()

for frame in root:
    frameNumber=int(frame.attrib['number'])+1
    pictureName='./images/'+workDir+format(frameNumber,'04')+'.jpg'
    print(pictureName)
    img1=cv2.imread(pictureName)
    cv2.imshow('image',img1)
    objectList=frame[0]
    for object in objectList:
        box=object[1]
        objectId=int(object.attrib['id'])
        if objectId==objectNum:
            if frameNumber>=startFrame and frameNumber<=endFrame:
                
                   
                x,y,w,h=float(box.attrib['xc']),float(box.attrib['yc']),float(box.attrib['w']),float(box.attrib['h'])
                plotArr=finarr(x,y,h,w,pictureName,rArr)
                shift=plotArr[0][0]
                D=plotArr[0][1]
                
                ind=D.index(max(D))
                shif=shift[ind]
                sig=0.1
                D=norm(D)
                if (frameCnt%10==0) :
                    print("completed")
                    plt.plot(shift,sigmoid(D,sig),'--')
                
                #similarity.append(D)

                for g in range(1,3):
                    shift1=plotArr[g][0]
                    D1=plotArr[g][1]
                    ind1=D1.index(max(D1))
                    if g==1:
                        shif1=shift1[ind1]
                    else :
                        shif2=shift1[ind1]
            xn=xp=x
            if (frameNumber>=14):
                if (shif<=10 and shif>=-10):
                    xp=xn=int(x+shif)
                else:
                    xn=xp
            
            print("the value of new x is ",xn)
            yn=y
            hn=h
            wn=w
            xn1=x+shif1
            xn2=x+shif2
            #cv2.rectangle(img1,(int(xn1-w/2),int(y-h/2)),(int(xn1+w/2),int(y+h/2)),(255,0,255),1)
            #cv2.rectangle(img1,(int(xn2-w/2),int(y-h/2)),(int(xn2+w/2),int(y+h/2)),(255,0,0),1)
            cv2.rectangle(img1,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,0,255),1)    
            cv2.rectangle(img1,(int(xn-w/2),int(y-h/2)),(int(xn+w/2),int(y+h/2)),(0,255,0),1)
            
        
            #cv2.rectangle(img1,(xn,yn),(xn+wn,yn+hn),(0,255,0),2)
            #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('image',img1)
            cv2.waitKey(75)
                
    frameCnt+=1
    
    

    #plt.legend(loc='best')
    #plt.show()
    #print(similarity)
    xn=x+shif
    
        
