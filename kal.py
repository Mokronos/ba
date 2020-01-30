from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

def gau(sigma, mu, x):
    #gaussian function
    y = (1/(sigma * (2*math.pi)**0.5)) * exp(-0.5*((x-mu)/sigma)**2)

    return y

print(math.pi)


