import numpy as np
import cv2 as cv2,cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
from helper import *
np.set_printoptions(threshold=sys.maxsize)


#load images:
#img1 = source
#img2 = target
img1 = cv2.imread("./box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./box_in_scene.png",cv2.IMREAD_GRAYSCALE)

#init orb detector
orb = cv2.ORB_create()

#find keypoints
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#create matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Match Descriptors
matches = bf.match(des1,des2)

#Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

#draw first 10 matches
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#img4 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
#print(kp1)
#print(kp1[0].pt)
#print(des1)
#print(des2)
#plt.imshow(img4)
#plt.show()

#plt.imshow(img3), plt.show()
#cv2.imshow("source_img",img1)
#cv2.imshow("target_img",img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#####################################################################################################
#define paths from which to load bbox data(class, confidence, coordiantes)
gt_path = "./data/1_58/groundtruth/bbox.txt"


#load bbox data: 1-gt, 2-det
cla1, conf1, bbox1, bboxx1 = read_bbox(gt_path)

#define first and last relevant frame aka first and last gt/detection
start, end = non_zero_index(cla1[:,0])

#coordinates of first bbox
xmin = int(bbox1[start, 1, 0])
ymin = int(bbox1[start, 1, 1])
xmax = int(bbox1[start, 1, 2])
ymax = int(bbox1[start, 1, 3])

#init first frame:
# -draw first detected bbox (either gt or detection)
# -define points to track
#(-define other features to track)
#--> save to VideoWriter
img = cv2.imread("./data/1_58/data/1_58" + str(start) + ".png",1)

img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0), 2)

img = cv2.circle(img,(xmin,ymin), 2, (0,0,255), 10)
img = cv2.circle(img,(xmax,ymax), 2, (0,0,255), 10)

#result.write(img)
img_old = img

#create orb detector
orb = cv2.ORB_create()
orb.setEdgeThreshold(1)

#define VideoWriter
result = cv2.VideoWriter("fmresultlinestest.avi",cv2.VideoWriter_fourcc(*"MJPG"),2,(1920+xmax-xmin,1080))

#crop image around object and make it grayscale
img_source = img[ymin:ymax,xmin:xmax,:]
img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)

#compute orb keypoints and descriptors of source image
kp1, des1 = orb.detectAndCompute(img_source, None)

#create matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

#define init points
p1 = (xmin,ymin) #top left
p2 = (xmax,ymin) #top right
p3 = (xmin,ymax) #bottom left
p4 = (xmax,ymax) #bottom right
#memory

#loop over frames start+1 --> end+1
for frame in range(start+1, end+1, 1):

    print(frame)
    #load frame from folder
    img_raw = cv2.imread("./data/1_58/data/1_58" + str(frame) + ".png",1)


    #fm prediction:
    #grayscale image
    #img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img_gray = img_raw
    #compute keypoints and descriptors of current frame
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    
    #compute matches between source and current frame descriptors
    matches = bf.match(des1,des2)
    
    #sort matches for smallest distance
    matches = sorted(matches, key = lambda x:x.distance)

    #draw bbox on frame
    img_gray = cv2.rectangle(img_gray, (int(bbox1[frame, 1, 0]),int(bbox1[frame, 1, 1])), (int(bbox1[frame, 1, 2]),int(bbox1[frame, 1, 3])), (0,255,0), 2)

    #draw matches on img
    img = cv2.drawMatches(img_source,kp1,img_gray,kp2,matches[:5], None, matchColor=(255,0,255),flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #img = cv2.drawMatches(img_source,kp1,img_gray,kp2,matches[:10], None, flags = cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)



    #for mat in matches:
    #    img2_idx = mat.trainIdx
    #    (x2,y2) = kp2[img2_idx].pt
    #    img = cv2.circle(img_gray,(int(x2),int(y2)), 2, (0,0,255), 2)



    #optical flow:
    #predict points in next frame
    


    #draw dot on top left corner and bottom right corner of bbox
    #img = cv2.circle(img,(int(bbox1[frame, 1, 0]),int(bbox1[frame, 1, 1])), 2, (0,0,255), 10)
    #img = cv2.circle(img,(int(bbox1[frame, 1, 2]),int(bbox1[frame, 1, 3])), 2, (0,0,255), 10)
    
    #draw optical flow points:

    #draw optical flow bbox from points:
    
    #test
    #write image onto video
    result.write(img)
    img_old = img_raw
    #print some stuff
    #print(frame)

#load, crop and grayscale image
img_test = cv2.imread("./data/1_58/data/1_58" + str(start) + ".png",1)
img_crop = img_test[ymin:ymax,xmin:xmax,:]
img_crop1 = np.copy(img_crop)
img_crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

#create orb detector
orb = cv2.ORB_create()
orb.setEdgeThreshold(1)
print(orb.getEdgeThreshold())

#compute points
kp,des = orb.detectAndCompute(img_crop_gray,None)


#draw points
#img_crop_kp = cv2.drawKeypoints(img_crop, kp, None, color=(0,0,255), flags=2)

#parameters for goodffeaturestotrack
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 2,
                       blockSize = 7 )

#calculate goodpointstotrack
ptt = cv2.goodFeaturesToTrack(img_crop_gray, mask = None, **feature_params)

for i in range(len(kp)):
    img_crop1 = cv2.circle(img_crop1, tuple(map(int,kp[i].pt)), 1, (255,0,255), -1)

#plt.imshow(img_crop1),plt.show()
#plt.figure()

img_crop2 = np.copy(img_crop)
ptt = np.int0(ptt)
for i in ptt:
    x,y = i.ravel()
    cv2.circle(img_crop2, (x,y), 1, (255,0,255), -1)


#show stuff
#plt.imshow(img_crop2), plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
result.release()
