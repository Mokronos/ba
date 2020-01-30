import cv2 as cv
import numpy as np
import sys
import time
import argparse
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("image", type=str, help= "path to input image file")
ap.add_argument("--image2", type=str, help= "path to input2 image file")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
img2 = cv.imread(args["image2"])
imgg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgg2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
#cv.imshow("img", img)
#cv.imshow("imgg", imgg)
#cv.imshow("imgg2", imgg2)

def o_flow(I1g, I2g, window_size, tau = 1e-5):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    
    w = window_size

    I1g = imgg/255.
    I2g = imgg2/255.

    fx = signal.convolve2d(I1g, kernel_x, boundary = "symm", mode = "same")
    fy = signal.convolve2d(I1g, kernel_y, boundary = "symm", mode = "same")
    ft = signal.convolve2d(I2g, kernel_t, boundary = "symm", mode = "same") + signal.convolve2d(I1g, -kernel_t, boundary = "symm", mode = "same")
    
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)

    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            
            b = np.reshape(It, (It.shape[0],1))
            A = np.vstack((Ix, Iy)).T
            print(i,j)    
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T,A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)
                u[i,j] = nu[0]
                v[i,j] = nu[1]
    return (u,v)
    

#cv.imshow("fx", fx)
#cv.imshow("fy", fy)
#cv.imshow("ft", ft)

#print(kernel_x)
#print(kernel_y)
#print(np.array(img).shape)
#print(np.array(imgg).shape)
#print(np.array(fx).shape)
#print(imgg)
#print(fx)

u, v = o_flow(imgg, imgg2, 3)

print(u)
print(v)


