import numpy as np
from multiprocessing import Process
from sklearn import mixture
import time
import os
import sys
import logging


# counts = [0]*256
# path1 = "d:/count.txt"
# path2 = "d:/gaussian.txt"

# def save_count(path,count):
#     file = open(path,'w')
#     for i in range(len(count)):
#         file.write(str(count[i])+'\n')
#     file.close()


# single gauss component
class Gauss:
    def __init__(self, weight, mean, covariance):
        self.weight_ = weight
        self.mean_ = mean
        self.covariance_ = covariance


# single block
class Block:
    def __init__(self):
        self.gauss_num_ = 0
        self.gausses_ = []

    def add_gauss(self, gauss):
        self.gausses_.append(gauss)

def save_block(path, block):
    file = open(path,'w')
    for i in range(len(block.gausses_)):
        file.write(str(block.gausses_[i].weight_)+' '+str(block.gausses_[i].mean_)+' '+str(block.gausses_[i].covariance_)+'\n')
    file.close()


# read all data into a array
def read_all_data(file_path, data):
    f = open(file_path, 'rb')
    filedata = f.read()
    filesize = f.tell()
    print(filesize)
    filedata2 = bytearray(filedata)
    for index in range(0, filesize):
        data[index] = filedata2[index]
    # data = bytearray(f.read())
    print("file size:"+str(filesize)+" byte(s)")
    f.close()


# train xth GMM
def train_single_block(index,
                       width,
                       depth,
                       width_num,
                       depth_num,
                       side,
                       ubg,
                       data):

    height_index = int(index / (width_num * depth_num))
    depth_index = int((index - height_index * (width_num * depth_num)) / width_num)
    width_index = int(index - height_index * (width_num * depth_num) - depth_index * width_num)

    start_width = width_index * side
    start_depth = depth_index * side
    start_height = height_index * side

    # print("--------IN BLOCK:"+str(index))
    # print("block num:"+str(width_num)+" "+str(depth_num))
    # print("block coordinates:"+str(width_index)+" "+str(depth_index)+" "+str(height_index))

    obs = [[]] * side * side * side
    zero = True
    zero_count = 0;
    for x in range(0, side):
        for y in range(0, side):
            for z in range(0, side):
                final_index = x * side * side + y * side + z
                data_index = (start_height + x) * width * depth + (start_depth + y) * width + (start_width + z)
                temp = data[data_index]
                # if index == 456:
                #     counts.append(temp)
                # if temp != 0:
                #     zero_count+=1
                #     zero = False
                obs[final_index] = [temp]
    # if zero == True:
    #     print("block:"+str(index)+" is zero")
    #     return Block()
    # print(str(index)+" is non-zero:"+str(zero_count))
    # if index == 456:
    #     save_count(path1,counts)

    final_component_num = 4
    g = mixture.GaussianMixture(n_components=final_component_num)
    g.fit(obs)
    final_g = g

    # max_bic = g.bic(np.array(obs))
    # max_num = min(ubg, len(obs))
    # for component_num in range(2, max_num+1):
    #     g = mixture.GaussianMixture(n_components=component_num)
    #     g.fit(obs)
    #     bic_temp = g.bic(np.array(obs))
    #     if index == 456:
    #         print component_num,bic_temp
    #     if bic_temp < max_bic:
    #         final_g = g
    #         final_component_num = component_num
    #         max_bic = bic_temp

    block = Block()
    block.gauss_num_ = final_component_num

    for component_index in range(0, final_component_num):
        gauss = Gauss(final_g.weights_[component_index], final_g.means_[component_index][0], final_g.covariances_[component_index][0][0])
        block.add_gauss(gauss)
    # if index == 456:
    #     save_block(path2,block)
    return block


# train a part of original data
def train_blocks(result_disk_address, data_source, block_num, index, stride, data, width, depth, depth_num, width_num, side, ubg):
    block_gmm = [Block()] * stride
    end_block = (index+1)*stride
    end_index = stride
    for i in range(0, stride):
        if index * stride + i >= block_num:
            end_block = index*stride+i;
            end_index = i
            break;
        block_gmm[i] = train_single_block(index * stride + i, width, depth, width_num, depth_num, side, ubg, data)

    gmm_output = result_disk_address + data_source + '_GMM_Result_'+str(index)+'.txt'

    # restore block_sgmm into txt file
    with open(gmm_output, "w") as f_out:
        for i in range(0, end_index):
            f_out.write(str(block_gmm[i].gauss_num_)+'\n')
            for j in range(0, block_gmm[i].gauss_num_):
                f_out.write(str(block_gmm[i].gausses_[j].weight_)+'\n')
                f_out.write(str(block_gmm[i].gausses_[j].mean_)+'\n')
                f_out.write(str(block_gmm[i].gausses_[j].covariance_)+'\n')

    print("-----------IN FILE:"+str(index)+" training and saving blocks from "+str(index*stride)+" to "+str(end_block)+" done")



if __name__ == '__main__':
    result_disk_address = ""
    data_source = ""
    width = 0
    depth = 0
    height = 0
    process_num = 0
    side = 0
    if len(sys.argv) == 1:
        result_disk_address = raw_input("input disk address:")
        data_source = raw_input('input the data name:')
        width = int(raw_input('weight:'))
        depth = int(raw_input('depth:'))
        height = int(raw_input('height:'))
        side = int(raw_input('side:'))
        process_num = int(raw_input('input the process num (must be the divisor of the block number):'))
    else:
        result_disk_address = sys.argv[1]
        data_source = sys.argv[2]
        width = int(sys.argv[3])
        depth = int(sys.argv[4])
        height = int(sys.argv[5])
        side = int(sys.argv[6])
        process_num = int(sys.argv[7])

    if not os.path.exists(result_disk_address + data_source + ".raw"):
        print('file doesn\'t exists')
        exit(0)

    print("disk address:" + result_disk_address)
    print("data name:" + data_source)
    print("width:" + str(width) + " depth:" + str(depth) + " height:" + str(height) + " side:" + str(side))
    print("process num (file num):" + str(process_num))


    ubg = 4
    np.random.seed(1)
    width_num = width / side
    depth_num = depth / side
    height_num = height / side
    total_num = width_num * depth_num * height_num
    data = [0] * width * depth * height
    stride = (total_num+process_num-1) / process_num
    print("stride:"+str(stride))

    read_all_data(result_disk_address + data_source + '.raw', data)
    logging.debug(data)
    begin_time = time.localtime(time.time())

    print("total_num = " + str(total_num))
    proc_record = []
    for i in range(0, process_num):
        p = Process(target=train_blocks, args=(result_disk_address,
                                               data_source,
                                               total_num,
                                               i,
                                               stride,
                                               data,
                                               width,
                                               depth,
                                               width_num,
                                               depth_num,
                                               side,
                                               ubg))
        p.start()
        proc_record.append(p)
    for p in proc_record:
        p.join()
    print("training GMM done.")
    print time.strftime('Training begin at %Y-%m-%d %H:%M:%S',begin_time)
    # train_single_block(73800)

