import utilities
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def least_square_linear(x,y):
    oneList = [1]*len(x)
    X = np.column_stack((oneList,x))
    Tr = X.T
    Inv = np.linalg.inv(Tr.dot(X))
    A = Inv.dot(Tr).dot(y)
    print(A)
    
    Error = np.sum(np.square((A[0]+(A[1]*x))-y))
    return Error

def least_square_polynomial(x,y):
    oneList = [1]*len(x)
    xS = np.square(x)
    xC = xS*x
    X = np.column_stack((oneList,x,xS,xC))
    Tr = X.T
    Inv = np.linalg.inv(Tr.dot(X))
    A = Inv.dot(Tr).dot(y)
    
    Error = np.sum(np.square((A[0]+(A[1]*x)+(A[2]*xS)+(A[3]*xC))-y))
    return Error

def least_square_sin(x,y):
    oneList = [1]*len(x)
    sinx = np.sin(x)
    X = np.column_stack((oneList,sinx))
    Tr = X.T
    Inv = np.linalg.inv(Tr.dot(X))
    A = Inv.dot(Tr).dot(y)

    Error = np.sum(np.square((A[0]+(A[1]*sinx))-y))
    return Error


def method_choice(x, y):
    lin = least_square_linear(x,y)
    pol = least_square_polynomial(x,y)
    sin = least_square_sin(x,y)
    method = lin
    
    if (sin < (0.9*method)):
        #print("sin")
        method = sin
    if (pol < (0.75*method)):
        #print("poly")
        method = pol

    #print("finished")    
    return method
    
def run():
    xs, ys = utilities.load_points_from_file(str(sys.argv[1]))
    xSplit = np.array_split(xs, (len(xs)/20))
    ySplit = np.array_split(ys, (len(ys)/20))
    total = 0
    utilities.view_data_segments(xs, ys)
    for i in range(0,len(xSplit)):
        total = total + method_choice(xSplit[i], ySplit[i])
    print(total)

    
if __name__ == "__main__":
    run()
