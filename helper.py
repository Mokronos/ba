import cv2 as cv
import numpy as np
import argparse
import sys
np.set_printoptions(threshold=sys.maxsize)

ap = argparse.ArgumentParser()
ap.add_argument("video", type=str, help = "input vid")
args = vars(ap.parse_args())

def cvi(path):
    v = cv.VideoCapture(path)
    mem = []
    f_total = int(v.get(cv.CAP_PROP_FRAME_COUNT))
    print(f_total)
    for i in range(0,f_total):
        ok , frame = v.read()
        mem.append(frame)

    return np.array(mem)

mem = cvi(args["video"])

for i in range(0, mem.shape[0]):
    print(mem[i,:,:,0])


print(mem.shape)
