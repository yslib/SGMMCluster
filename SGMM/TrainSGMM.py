import numpy as np
from multiprocessing import Process
from sklearn import mixture
import time
import os
import sys

# volume data order: width > depth > height
# block order: width > depth > height
# version 1.0 only for int raw data
# by gxchen


# single gauss component
class Gauss:
    def __init__(self, weight, mean, covariance):
        self.weight_ = weight
        self.mean_ = mean  # 3D array
        self.covariance_ = covariance  # 9D array


# single bin in histogram
class Bin:
    def __init__(self, probability, gauss_count , sgmm):
        self.probability_ = probability
        self.gauss_count_ = gauss_count
        self.gausses_ = []

    def add_gauss(self, gauss):
        self.gausses_.append(gauss)


# single block
class Block:
    def __init__(self):
        self.bin_num_ = 0
        self.bins_ = []
        self.bin_indexs_ = []

    def add_bin(self, bin):
        self.bins_.append(bin)


# data_source = 'Combustion'
# width = 480
# depth = 720
# height = 112
# process_num = 4
# disk_address = 'c:/train/'
#
# src_raw_name = disk_address+data_source+'.raw'
# side = 16
# zero_block_threshold = 0.003
# block_size = side * side * side
# width_num = width / side
# depth_num = depth / side
# height_num = height / side
#
# total_num = width_num * depth_num * height_num
# max_bin_num = 128
# ubg = 4  # max component number
# restore_raw = bytearray(width * depth * height)
# np.random.seed(1)
# stride = total_num / process_num
#
# f_all_data = open(src_raw_name, 'rb')
# f_all_data.seek(0, 0)
# all_data = bytearray(f_all_data.read())
# all_hit = [0] * width * depth * height


# read index th block data
def read_block(index,all_data, width, depth, width_num, depth_num, block_size,side):
    height_index = index / (width_num * depth_num)
    depth_index = (index - height_index * width_num * depth_num) / width_num
    width_index = index - height_index * width_num * depth_num - depth_index * width_num

    result_data = [0] * block_size
    for z in range(0, side):  # width
        for y in range(0, side):  # depth
            for x in range(0, side):  # height
                final_index_z = height_index * side + z
                final_index_y = depth_index * side + y
                final_index_x = width_index * side + x
                final_index = final_index_z * width * depth + final_index_y * width + final_index_x
                result_data[z * side * side + y * side + x] = all_data[final_index]

    return result_data


# train index th block data
def train_single_block(block_index,
                       block_data,
                       block_size,
                       max_bin_num,
                       side,
                       ubg):
    block = Block()
    count = [0] * max_bin_num
    train_data = [] * max_bin_num
    for i in range(0, max_bin_num):
        train_data.append([])

    non_zero_count = 0
    for z in range(0, side):
        for y in range(0, side):
            for x in range(0, side):
                final_index = z * side * side + y * side + x
                index = block_data[final_index] / 2
                count[index] += 1  # map to value-distribution
                train_data[index].append([x, y, z])
                if block_data[final_index] != 0:
                    non_zero_count += 1

    # train SGMM
    block.bin_num_ = 0
    if non_zero_count > int(side * side * side * 0.3):  # make sure not a empty block
        for bin_index in range(0, max_bin_num):
            if count[bin_index] > 0:
                block.bin_indexs_.append(bin_index)
                block.bin_num_ += 1

        for bin_count in range(0, block.bin_num_):
            real_index = block.bin_indexs_[bin_count]
            # if train_data[i] is empty or very small, skip it
            if len(train_data[real_index]) <= 0:  # not happen when equal 0, you can make it larger to speed up
                continue

            g = mixture.GaussianMixture(n_components=1, tol=1e-5, max_iter=5000)
            g.fit(train_data[real_index])
            max_bic = g.bic(np.array(train_data[real_index]))
            final_g = g
            final_component_num = 1

            max_num = min(ubg, len(train_data[real_index]))
            for component_num in range(2, max_num+1):
                    g = mixture.GaussianMixture(n_components=component_num, tol=1e-5, max_iter=5000)
                    g.fit(train_data[real_index])
                    bic_temp = g.bic(np.array(train_data[real_index]))
                    if block_index == 456:
                        print component_num,bic_temp
                    if bic_temp < max_bic:
                        final_g = g
                        final_component_num = component_num
                        max_bic = bic_temp

            # already got final SGMM for bin i
            bin = Bin(1.0 * count[real_index]/block_size, final_component_num, final_g)
            for i in range(0, final_component_num):
                gauss = Gauss(final_g.weights_[i], final_g.means_[i], final_g.covariances_[i])
                bin.add_gauss(gauss)
            block.add_bin(bin)

    print("training block index " + str(block_index)+" done, bin_num_ = "+str(block.bin_num_))

    return block


# make sure the value if not to small, else it will result in wrong input in C++ program
def check_value(value_in):
    if value_in < 1.0e-40:
        return 1.0e-40
    else:
        return value_in


# train a part of original data
# and save sgmm arguments into a txt file
def train_blocks(disk_address,
                 data_source,
                 block_num,
                 index,
                 stride,
                 src_raw_name,
                 all_data,
                 width,
                 depth,
                 width_num,
                 depth_num,
                 max_bin_num,
                 block_size,
                 side,
                 ubg):
    block_sgmm = [Block()] * stride
    end_block = (index+1)*stride
    end_index = stride
    with open(src_raw_name, 'rb') as f_src:
        for i in range(0, stride):
            if index*stride + i >= block_num:
                end_block = index*stride+i
                end_index = i
                break
            block_data = read_block(index * stride + i,all_data,width, depth, width_num, depth_num, block_size,side)
            block_sgmm[i] = train_single_block(index * stride + i, block_data, block_size, max_bin_num, side, ubg)

    sgmm_output = disk_address + data_source + '_SGMM_Result_'+str(index)+'.txt'  # only sgmm arguments

    # restore block_sgmm into txt file
    with open(sgmm_output, "w") as f_out:
        for i in range(0, end_index):
            # f_out.write(str(index * stride + i) + '###\n')  # test only
            idx = index*stride+i
            if idx == 20 or idx == 13 or id == 6 or idx == 0:
                print("block_index:"+str(idx)+" bin num:"+str(block_sgmm[i].bin_num_))
            f_out.write(str(block_sgmm[i].bin_num_) + '\n')
            for bin_count in range(0, block_sgmm[i].bin_num_):
                real_bin_index = block_sgmm[i].bin_indexs_[bin_count]
                f_out.write(str(real_bin_index)+' ' + str(check_value(block_sgmm[i].bins_[bin_count].probability_))+' '+str(block_sgmm[i].bins_[bin_count].gauss_count_)+'\n')
                for k in range(0, block_sgmm[i].bins_[bin_count].gauss_count_):
                    f_out.write(str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].weight_))+'\n')
                    f_out.write(str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].mean_[0]))+' '+str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].mean_[1]))+' '+str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].mean_[2]))+'\n')
                    f_out.write(str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].covariance_[0][0]))+' '+str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].covariance_[0][1]))+' '+str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].covariance_[0][2]))+'\n')
                    f_out.write(str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].covariance_[1][1]))+' '+str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].covariance_[1][2]))+'\n')
                    f_out.write(str(check_value(block_sgmm[i].bins_[bin_count].gausses_[k].covariance_[2][2]))+'\n')

    print("----------IN FILE:"+str(index)+" training and saving blocks from "+str(index*stride)+" to "+str(end_block)+" done")


# train all block, parallel computing, assign into 4 cpu kernel
if __name__ == '__main__':
    disk_address =""
    data_source = ""
    width = 0
    depth = 0
    height = 0
    process_num = 0
    side =0
    if len(sys.argv) == 1:
        disk_address = raw_input("input disk address:")
        data_source = raw_input('input the data name:')
        width = int(raw_input('weight:'))
        depth = int(raw_input('depth:'))
        height = int(raw_input('height:'))
        side = int(raw_input('side:'))
        process_num = int(raw_input('input the process num (must be the divisor of the block number):'))
    else:
        disk_address = sys.argv[1]
        data_source=sys.argv[2]
        width = int(sys.argv[3])
        depth = int(sys.argv[4])
        height = int(sys.argv[5])
        side = int(sys.argv[6])
        process_num = int(sys.argv[7])

    if not os.path.exists(disk_address+data_source+".raw"):
        print('file doesn\'t exists')
        exit(0)

    print("disk address:"+disk_address)
    print("data name:"+data_source)
    print("width:"+str(width)+" depth:"+str(depth)+" height:"+str(height)+" side:"+str(side))
    print("process num (file num):"+str(process_num))

    src_raw_name = disk_address+data_source+'.raw'

    zero_block_threshold = 0.003
    block_size = side * side * side

    width_num = width / side
    depth_num = depth / side
    height_num = height / side

    total_num = width_num * depth_num * height_num

    max_bin_num = 128
    ubg = 4  # max component number
    restore_raw = bytearray(width * depth * height)
    np.random.seed(1)
    stride = (total_num+process_num-1) / process_num
    f_all_data = open(src_raw_name, 'rb')
    f_all_data.seek(0, 0)
    all_data = bytearray(f_all_data.read())
    all_hit = [0] * width * depth * height
    begin_time = time.localtime(time.time())
    cpu_time_begin = time.clock()
    proc_record = []
    for i in range(0, process_num):  # a block / 3 seconds
        p = Process(target=train_blocks, args=(disk_address,
                                               data_source,
                                               total_num,
                                               i,
                                               stride,
                                               src_raw_name,
                                               all_data,
                                               width,
                                               depth,
                                               width_num,
                                               depth_num,
                                               max_bin_num,
                                               block_size,
                                               side,
                                               ubg))
        p.start()
        proc_record.append(p)
    
    for p in proc_record:
        p.join()
    
    print("training SGMM done.")
    cpu_time_end = time.clock();
    print time.strftime('Training began at %Y-%m-%d %H:%M:%S', begin_time)
    print time.strftime('Training finished at %Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    print("cpu time cost in python :"+str(cpu_time_end-cpu_time_begin)+"s.")
    # with open(src_raw_name, "rb") as f_src:
    #     single_block_data = read_block(3500)
    #     train_single_block(3500, single_block_data)












