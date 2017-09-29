from multiprocessing import Process
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import numpy as np
import os
import math
import sys

def my_test(index):
    print("this is process:"+str(index))


if __name__ == '__main__':
    print("asdf")
    proc =[]
    process_num = 5
    for i in range(0, process_num):
        p = Process(target=my_test, args=(i,))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()



