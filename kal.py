from scipy import signal
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from filterpy.kalman import KalmanFilter
import helper as h
import cv2
import numpy.ma as ma


np.set_printoptions(precision = 4, suppress=True)



'''
# inputs:
# - measurement(z)
# - measurement uncertainty(r)
# - initial measurement
# - initial measurement uncertainty (variance = standard deviation ** 2)
#
# outputs:
# - system state estimate(x)
# - estimate uncertainty(p) (cov)
#
# parameters:
# - kalman gain(k)
# - time_step(d_t)

# x: System State
# u: Control Variable/ Sytem Input (measurable(deterministic) Input to state, e.g. Force)
# w: System Noise (unmeasurable Input to state)
# v: Measurement Noise
# z: Measurement
# H: Observation Matrix
# F: State Transition Matrix
# G: Control Matrix
# P: Estimate Uncertainty Covariance Matrix
# B: Input Matrix(?)
# Q: Process Noise Matrix
# K: Kalman Gain Matrix
# R: Measurement Uncertainty

'''
#takes kalman filter object, current state, next measurement and parameters --> updates state with kalman filter
# TODO make it read parameters out of parapath(define dimensions, deltat, matrices) --> let clip run with different parameters if txt file not already there (give parameter combination nickname, parameters themselfs is too long)
def kalclip(bbox, gt, representation, R = 500, Q = 500, P = 1):
    
    #define start(no need to initiate before detections) 
    start = h.getstartindex(bbox)

    #for 30fps --> 1/fps
    deltat = 0.033333333333333333333
    
    #define dimensions and model(depends on representation)
    if representation == "cor": 
        n_x, n_z = 4,4
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z)
        Kal.F = np.eye(n_x)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0][0] = Kal.H[1][1] = Kal.H[2][2] = Kal.H[3][3] = 1
        init = np.zeros((n_x, 1))
        init[:,0] = gt[start-1,:]

    elif representation == "cen": 
        n_x, n_z = 4,4
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z)
        Kal.F = np.eye(n_x)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0][0] = Kal.H[1][1] = Kal.H[2][2] = Kal.H[3][3] = 1
        init = np.zeros((n_x, 1))
        init[:,0] = gt[start-1,:]

    elif representation == "asp": 
        n_x, n_z = 7,4
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z)
        Kal.F = np.eye(n_x)
        Kal.F[0,4] = Kal.F[1,5] = Kal.F[2,6] = deltat
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0,0] = Kal.H[1,1] = Kal.H[2,2] = Kal.H[3,3] = 1
        init = np.zeros((n_x, 1))
        init[:4,0] = gt[start-1,:]
        init[4:,0] = [300,-300,100]

    #parameters to define(try these on all representations)

    Kal.R = np.eye(n_z)  * R
    Kal.Q = np.eye(n_x) * Q
    Kal.P *= P


    Kal.x = init

    #create new array for results of the algorithm
    mem = ma.array(np.zeros_like(bbox), mask = True)

    for i in range(start, bbox.shape[0],1):
        z = bbox[i,:]
        if bbox.mask[i,0] == True:
            z = None
        Kal.predict()
        Kal.update(z)
        x = Kal.x
        y = x[:4]
        
        #write into resulting array
        for j in range(len(y)):
            mem[i,j] = y[j]

    #just return result --> do back-transformation into other representations outside of this function(as well as creating visuals)
    return mem
