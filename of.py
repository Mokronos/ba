import numpy as np
import cv2
from helper import *

#define paths from which to load bbox data(class, confidence, coordiantes)
gt_path = "./data/1_58/groundtruth/bbox.txt"

#define VideoWriter
result = cv2.VideoWriter("ofresult99.avi",cv2.VideoWriter_fourcc(*"MJPG"),2,(1920,1080))

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

result.write(img)
img_old = img

#idea:
#track corner + middle of bbox --> take (log) average coordinates of corner points as corners for bbox?
# and use goodpoints to track on first bbox --> track those --> how to define bbox???


#optical flow init:
#parameters
lk_params = dict( winSize  = (40,40),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#define init points
p1 = (xmin,ymin) #top left
p2 = (xmax,ymin) #top right
p3 = (xmin,ymax) #bottom left
p4 = (xmax,ymax) #bottom right
#memory
pts_to_track = 2 #number of points to track
mem_of = np.zeros((len(cla1),pts_to_track,1,2), dtype = np.float32)
mem_of[start,:,:,:] = np.array([[p1],[p4]])

#loop over frames start+1 --> end+1
for frame in range(start+1, end+1, 1):

    #load frame from folder
    img_raw = cv2.imread("./data/1_58/data/1_58" + str(frame) + ".png",1)


    #optical flow:
    #predict points in next frame
    mem_of[frame,:,:,:], st, err = cv2.calcOpticalFlowPyrLK(img_old, img_raw, mem_of[frame-1,:,:,:], None, **lk_params) 

    #draw bbox on frame
    img = cv2.rectangle(img_raw, (int(bbox1[frame, 1, 0]),int(bbox1[frame, 1, 1])), (int(bbox1[frame, 1, 2]),int(bbox1[frame, 1, 3])), (0,255,0), 2)

    #draw dot on top left corner and bottom right corner of bbox
    #img = cv2.circle(img,(int(bbox1[frame, 1, 0]),int(bbox1[frame, 1, 1])), 2, (0,0,255), 10)
    #img = cv2.circle(img,(int(bbox1[frame, 1, 2]),int(bbox1[frame, 1, 3])), 2, (0,0,255), 10)
    
    #draw optical flow points:
    img = cv2.circle(img,tuple(mem_of[frame, 0,0, :]), 2, (255,0,255), 10)
    img = cv2.circle(img,tuple(mem_of[frame, 1,0, :]), 2, (255,0,255), 10)

    #draw optical flow bbox from points:
    img = cv2.rectangle(img, tuple(mem_of[frame, 0,0, :]),tuple(mem_of[frame, 1,0, :]), (255,255,0), 2)

    #write image onto video
    result.write(img)
    img_old = img_raw
    #print some stuff
    #print(frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
result.release()
