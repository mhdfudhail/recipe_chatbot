import cv2
import cv2.aruco as aruco
import numpy as np
import time


def findArucoMarker(img, markerSize=6, totalMarker=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters()
    bbx, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

    if draw:
        aruco.drawDetectedMarkers(img, bbx)

    return [bbx, ids]

def arucoPositionTarget(img, marker, arucoId): #enemy centroid
    print(f"vehicle Id: {arucoId}")
    tx,ty,bx,by = marker[arucoId][0],marker[arucoId][1],marker[arucoId][4],marker[arucoId][5]
    t1,t2 = marker[arucoId][2],marker[arucoId][3]
    cv2.circle(img, (t1,t2),3, (0,0,255),3)
    cv2.circle(img, (tx,ty),3, (0,0,255),3)
    # print(f"X:{tx}, Y:{ty}")
    # find centroid of aruco marker
    cx = int((tx+bx)//2)
    cy = int((ty+by)//2)
    # print(f"cx:{cx}, cy:{cy}")
    cv2.circle(img, (cx,cy),2, (255,0,0),2)

def main():
    frameWidth = 1280
    frameHeight = 720
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    markerDict = {}

    if(cap.isOpened()): 
        while True:
            ret, img = cap.read() 
            arucoFound = findArucoMarker(img)
            # print(arucoFound)
            if len(arucoFound[0])!=0:
                for bbx, ids in zip(arucoFound[0], arucoFound[1]):
                    idCord = [bbx]
                    # print(f"full cord: {idCord}")
                    # print("fist:",bbx[0][0])
                    # print("second:",bbx[0][1])
                    idNum = ids[0]
                    idCord = [int(bbx[0][0][0]),
                                int(bbx[0][0][1]),
                                int(bbx[0][1][0]),
                                int(bbx[0][1][1]),
                                int(bbx[0][2][0]),
                                int(bbx[0][2][1])]
                    # print(idCord)
                    marker={idNum:idCord}
                    markerDict.update(marker)

            print(markerDict)
            # for key in list(markerDict.keys()):
            #         if key not in [id[0] for id in arucoFound[1]]:
            #             del markerDict[key]

            cv2.imshow('img', img)
             
            if cv2.waitKey(30) & 0xff == ord('q'): 
                break
                
        cap.release() 
        cv2.destroyAllWindows() 
    else: 
        print("Camera disconnected") 

if __name__== '__main__':
    main()