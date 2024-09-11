import numpy as np
import pandas as pd
import math


    
def triplet_prediction_accuracy(distances1,distances2,distances3,types=None,mode="triplet"):
    # distances1: anc and pos
    # distances2: anc and neg
    # distances3: pos and neg
    N = len(distances1)
    distances1 = np.array(distances1)
    distances2 = np.array(distances2)
    distances3 = np.array(distances3)

    
    c1 = distances2-distances1
    c2 = distances3-distances1
    n = 0
    if types==None:
        for i in range(N):
            if c1[i] > 0 and c2[i] > 0:
                n+=1
        acc = n/N
        return acc
    else:
        s1,s2,s3,N1,N2,N3=0,0,0,0,0,0
        for i in range(len(c1)):
            if types[i] == "ONE_CLASS_TRIPLET":
                N1 += 1
            elif types[i] == "TWO_CLASS_TRIPLET":
                N2 += 1
            elif types[i] == "THREE_CLASS_TRIPLET":
                N3 += 1
            if c1[i] > 0 and c2[i] > 0:
                n+=1
                if types[i]=="ONE_CLASS_TRIPLET":
                    s1+=1
                elif types[i]=="TWO_CLASS_TRIPLET":
                    s2+=1
                elif types[i]=="THREE_CLASS_TRIPLET":
                    s3+=1
        acc = n/N
        acc1 = s1/N1
        acc2 = s2/N2
        acc3 = s3/N3
        return acc,acc1,acc2,acc3




