from scipy import signal
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from filterpy.kalman import KalmanFilter
import helper as h
import cv2
import numpy.ma as ma
import of


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
def kalclip(bbox, gt, model,clippath, stats, ofpoint = 0, R = 500, Q = 500, P = 0):
    
    #define start(no need to initiate before detections) 
    start = h.getstartindex(bbox)
    
    clipname = h.extract(clippath)
    #for 30fps --> 1/fps
    #irrelevant in my case --> set 1 (i dont rly make a reference to reality here)
    deltat = 1
    
    #define what model has what representation for retransformation after:
    correp = []
    cenrep = ["cenof","simplecen"]
    asprep = ["aspof","aspofwidth","aspofwidthman"]
    std = 70

    #define dimensions and model(depends on representation)
    if model == "simplecen": 

        #transform representation
        bbox = h.corcen(bbox)
        gt = h.corcen(gt)

        #set parameters
        n_x, n_z = 4,4
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z)
        Kal.F = np.eye(n_x)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0][0] = Kal.H[1][1] = Kal.H[2][2] = Kal.H[3][3] = 1
        Kal.Q = np.eye(n_x) * std**2
        for i in range(len(stats)):
            Kal.R[i,i] = stats[0,1,i] ** 2

        #init
        init = np.zeros(n_x)
        init[:] = gt[0,:]
        memp = np.zeros((bbox.shape[0],n_x))
        memk = np.zeros((bbox.shape[0],n_x))

    elif model == "cenof": 
        #transform representation
        bbox = h.corcen(bbox)
        gt = h.corcen(gt)

        #set parameters
        n_x, n_z, n_u = 4,4,2
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z, dim_u = n_u)
        Kal.R = np.eye(n_z)  * R
        for i in range(len(stats)):
            Kal.R[i,i] = stats[1,1,i] ** 2
        Kal.Q = np.eye(n_x) * std**2 
        Kal.Q[0,0], Kal.Q[1,1]= stats[3,1,0], stats[3,1,1]
        Kal.P *= P
        Kal.F = np.eye(n_x)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0,0] = Kal.H[1,1] = Kal.H[2,2] = Kal.H[3,3] = 1
        Kal.B = np.eye(n_x, n_u)
        
        #init
        init = np.zeros(n_x)
        init[:4] = gt[0,:]
        memp = np.zeros((bbox.shape[0],n_x))
        memk = np.zeros((bbox.shape[0],n_x))

    elif model == "aspof": 
        #transform representation
        bbox = h.corasp(bbox)
        gt = h.corasp(gt)

        #set parameters
        n_x, n_z, n_u = 4,4,2
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z, dim_u = n_u)
        Kal.R = np.eye(n_z)  * R
        for i in range(len(stats)):
            Kal.R[i,i] = stats[2,1,i] ** 2
        Kal.Q = np.eye(n_x) * std**2
        Kal.Q[0,0], Kal.Q[1,1], Kal.Q[3,3]= stats[3,1,0] **2, stats[3,1,1] **2, 0.0001
        Kal.P *= P
        Kal.F = np.eye(n_x)
        Kal.B = np.eye(n_x, n_u)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0,0] = Kal.H[1,1] = Kal.H[2,2] = Kal.H[3,3] = 1

        #init
        init = np.zeros(n_x)
        init[:4] = gt[0,:]
        memp = np.zeros((bbox.shape[0],n_x))
        memk = np.zeros((bbox.shape[0],n_x))

    elif model == "aspofwidth": 
        #transform representation
        bbox = h.corasp(bbox)
        gt = h.corasp(gt)

        #set parameters
        n_x, n_z, n_u = 4,4,3
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z, dim_u = n_u)
        Kal.R = np.eye(n_z)  * R
        for i in range(len(stats)):
            Kal.R[i,i] = stats[2,1,i] ** 2
        Kal.Q = np.eye(n_x) * std**2
        #TODO put in stats for std width from of 
        Kal.Q[0,0], Kal.Q[1,1], Kal.Q[2,2], Kal.Q[3,3]= stats[3,1,0] **2, stats[3,1,1] **2, stats[3,1,2] ** 2, 0.0001
        Kal.P *= P
        Kal.F = np.eye(n_x)
        Kal.B = np.eye(n_x, n_u)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0,0] = Kal.H[1,1] = Kal.H[2,2] = Kal.H[3,3] = 1

        #init
        init = np.zeros(n_x)
        init[:4] = gt[0,:]
        memp = np.zeros((bbox.shape[0],n_x))
        memk = np.zeros((bbox.shape[0],n_x))



    elif model == "aspofwidthman": 
        #transform representation
        bbox = h.corasp(bbox)
        gt = h.corasp(gt)

        #set parameters
        n_x, n_z, n_u = 4,4,3
        Kal = KalmanFilter(dim_x = n_x, dim_z = n_z, dim_u = n_u)
        Kal.R = np.eye(n_z)  * R
        for i in range(len(stats)):
            Kal.R[i,i] = stats[2,1,i] ** 2
        Kal.Q = np.eye(n_x) * std**2
        Kal.Q[0,0], Kal.Q[1,1], Kal.Q[2,2], Kal.Q[3,3]= stats[3,1,0], stats[3,1,1], 1**2, 0.0001
        Kal.P *= P
        Kal.F = np.eye(n_x)
        Kal.B = np.eye(n_x, n_u)
        Kal.H = np.zeros((n_z,n_x))
        Kal.H[0,0] = Kal.H[1,1] = Kal.H[2,2] = Kal.H[3,3] = 1

        #init
        init = np.zeros(n_x)
        init[:4] = gt[0,:]
        memp = np.zeros((bbox.shape[0],n_x))
        memk = np.zeros((bbox.shape[0],n_x))


   
    #parameters to define(try these on all representations)

    Kal.x = init

    #create new array for results of the algorithm
    mem = ma.array(np.zeros_like(bbox), mask = True)
    mem[0,:] = init
    for j in range(memp.shape[1]):
        memp[0,j] = Kal.P[j,j]
    for j in range(memk.shape[1]):
        memk[0,j] = Kal.K[j,j]


    for i in range(1, bbox.shape[0],1):
        
        z = bbox[i,:]
        if bbox.mask[i,0] == True:
            z = None

        
        if model == "cenof":

            vx = ofpoint[i,0] - ofpoint[i-1,0]
            vy = ofpoint[i,1] - ofpoint[i-1,1]
            u = np.array([vx,vy])
            Kal.predict(u)
        elif model == "aspof":

            vx = ofpoint[i,0] - ofpoint[i-1,0]
            vy = ofpoint[i,1] - ofpoint[i-1,1]
            u = np.array([vx,vy])
            Kal.predict(u)
        elif model == "aspofwidth":

            vx = ofpoint[i,0] - ofpoint[i-1,0]
            vy = ofpoint[i,1] - ofpoint[i-1,1]
            vwidth = ofpoint[i,2] - ofpoint[i-1,2]
            u = np.array([vx,vy,vwidth])
            Kal.predict(u)
        elif model == "aspofwidthman":

            vx = ofpoint[i,0] - ofpoint[i-1,0]
            vy = ofpoint[i,1] - ofpoint[i-1,1]
            vwidth = ofpoint[i,2] - ofpoint[i-1,2]
            u = np.array([vx,vy,vwidth])
            Kal.predict(u)
        else:
            Kal.predict()
        Kal.update(z)
        x = Kal.x
        y = x[:4]
        
        #write into resulting array
        mem[i,:] = y[:4]

        for j in range(memp.shape[1]):
            memp[i,j] = Kal.P[j,j]
        for j in range(memk.shape[1]):
            memk[i,j] = Kal.K[j,j]



    #just return result --> do back-transformation into other representations outside of this function(as well as creating visuals)
    if model in correp:
        mem = mem
    elif model in cenrep:
        mem = h.cencor(mem)
    elif model in asprep:
        mem = h.aspcor(mem)
    return mem, memp, memk
