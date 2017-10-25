from multiprocessing import Process
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import numpy as np
import os
import math
import sys
import time

def my_test(index):
    print("this is process:"+str(index))


if __name__ == '__main__':
    begin_time = time.time();

    print time.strftime('Training done at %Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print time.strftime('Training costs: %Y-%m-%d %H:%M:%S', time.localtime(end_time - begin_time))
    end_time = time.time()
    print (end_time-begin_time);


