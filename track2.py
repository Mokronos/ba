import cv2 as cv
import numpy as np
import sys
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
args = vars(ap.parse_args())

out = cv.VideoWriter('trackresult.mp4', -1, 30, (1920,1080))
tracker = cv.TrackerCSRT_create()

v = cv.VideoCapture(args["video"])
fc = int(v.get(cv.CAP_PROP_FRAME_COUNT))

ok, frame = v.read()
bbox =  cv.selectROI(frame,False)

ok = tracker.init(frame, bbox)
mem = []
f = 0
while f<fc:
    ok, frame = v.read()

    ok, bbox = tracker.update(frame)
    
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame,p1,p2,(255,0,0),2,1)
    else:
        cv.putText(frame, "Tracking failed", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    cv.putText(frame, "frame: " + str(int(f)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    out.write(frame)
    mem.append(frame)
    f+=1

f = 0
cv.imshow("track",mem[f])

out.release()
while True:
    
    k = cv.waitKey(0)
    if k ==ord("k"):
        f+=1
    if k ==ord("j"):
        f-=1
    if k ==ord("l"):
        f+=50
    if k ==ord("h"):
        f-=50
    if k ==27:
        sys.exit()
    if 0 <= f <fc:
        
        cv.imshow("track",mem[f])
    
        





