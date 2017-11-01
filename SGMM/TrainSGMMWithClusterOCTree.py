from multiprocessing import Process
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import numpy as np
import os
import math
import sys

# single gauss component
class Gauss:
    def __init__(self, weight, mean, precision, determinant):
        self.weight_ = weight
        self.mean_ = mean  # 3D array
        self.precision_ = precision  # 9D array
        self.determinant_ = determinant


# single bin in histogram
class Cluster:
    def __init__(self, probability, gauss_count, sample_value):
        self.probability_ = probability
        self.gauss_count_ = gauss_count
        self.sample_value_ = sample_value
        self.gausses_ = []

    def add_gauss(self, gauss):
        self.gausses_.append(gauss)


# single block
class Block:
    def __init__(self):
        self.clusters_ = []
        self.cluster_num_ = 1

    def add_cluster(self, cluster):
        self.clusters_.append(cluster)

class Point3d:
    def __init__(self, xx = 0, yy = 0,zz = 0):
        self.x = xx
        self.y = yy;
        self.z = zz;



# width = 480
# depth = 720
# height = 112
# data_source = ''
# result_disk_address = 'c:/train/'
# process_num = 16  # make sure it is a divisor of block number
#
# side = 16
# zero_block_threshold = 0.003
# ubg = 4  # max component number
# cluster_weights = [1.0, 1.0, 1.0, 1.0]  # weights when clustering
# src_raw_name = result_disk_address + data_source+'.raw'
# restored_raw_name = result_disk_address+data_source+'_restored.raw'
# result_raw_name = result_disk_address+data_source+'_SGMM_Result.raw'  # restored data
# training_done_signal = result_disk_address + 'Training_Done.txt'
#
# block_size = side * side * side
# width_num = width / side
# depth_num = depth / side
# height_num = height / side
# total_num = width_num * depth_num * height_num
# restore_raw = bytearray(width * depth * height)
# np.random.seed(1)
# stride = total_num / process_num
#
# f_all_data = open(src_raw_name, 'rb')
# f_all_data.seek(0, 0)
# all_data = bytearray(f_all_data.read())
# all_hit = [0] * width * depth * height


# read index th block data
# def read_block(index, all_data, width, depth, width_num, depth_num,block_size,side):
#     height_index = index / (width_num * depth_num)
#     depth_index = (index - height_index * width_num * depth_num) / width_num
#     width_index = index - height_index * width_num * depth_num - depth_index * width_num
#
#     result_data = [0] * block_size
#     for z in range(0, side):  # width
#         for y in range(0, side):  # depth
#             for x in range(0, side):  # height
#                 final_index_z = height_index * side + z
#                 final_index_y = depth_index * side + y
#                 final_index_x = width_index * side + x
#                 final_index = final_index_z * width * depth + final_index_y * width + final_index_x
#                 result_data[z * side * side + y * side + x] = all_data[final_index]
#
#     return result_data

def read_block(min_point,max_point, all_data, width, depth):
    width_length = max_point.x - min_point.x
    depth_length = max_point.y - min_point.y
    height_lenght = max_point.z - min_point.z

    block_size = width_length*depth_length*height_lenght

    result_data = [0]*block_size

    for z in range(0,height_lenght):
        for y in range(0,depth_length):
            for x in range(0,width_length):
                final_index_z = min_point.z+z
                final_index_y = min_point.y+y
                final_index_x = min_point.x+x
                final_index = final_index_x + final_index_y*width+final_index_z*depth*width
                result_data[x+y*width_length+z*width_length*depth_length] = all_data[final_index]

    return result_data


def CH(k_means, train_data):
    means_of_all_data = np.mean(train_data, axis=0)
    k = len(k_means.cluster_centers_)
    n = len(train_data)
    count = [0]*k
    for i in range(0,n):
        count[k_means.labels_[i]]+=1
    traceB = 0
    for i in range(0,k):
        traceB += count[i]*np.sum(np.square(means_of_all_data-k_means.cluster_centers_[i]))
    traceW = 0
    for i in range(0,k):
        for j in range(0,n):
            traceW += np.sum(np.square((k_means.cluster_centers_[i]-train_data[j])))
    return (traceB/(k-1))/(traceW/(n-k))

# train index th block data
# def train_single_block(block_index, block_data, max_cluster_num, block_size, side, cluster_weights, ubg):
#     # clustering by kmeans
#
#     cluster_data = []
#     count = [0] * max_cluster_num
#     non_zero_count = 0
#     for z in range(0, side):
#         for y in range(0, side):
#             for x in range(0, side):
#                 index = z * side * side + y * side + x
#                 cluster_data.append([cluster_weights[0]*x, cluster_weights[1]*y, cluster_weights[2]*z, cluster_weights[3] * block_data[index]])
#                 if block_data[index] != 0:
#                     non_zero_count += 1
#
#
#     # here need a new standard to check the cluster number!
#
#
#     # check if zero_count too big
#     if non_zero_count > int(side * side * side * 0.003):
#         # print('here3')
#         # kmeans = KMeans(n_clusters=2, max_iter=1000).fit(cluster_data)
#         # final_kmeans = kmeans
#         # final_cluster_num = 2
#         # label = kmeans.labels_
#         # max_score = silhouette_score(cluster_data, label, metric='euclidean')
#         # # max_score = CH(kmeans, cluster_data)
#         # for cluster_index_num in range(3, max_cluster_num+1):
#         #     print('here4',cluster_index_num)
#         #     kmeans = KMeans(n_clusters=cluster_index_num, max_iter=1000).fit(cluster_data)
#         #     label = kmeans.labels_
#         #     present_score = silhouette_score(cluster_data, label, metric='euclidean')
#         #     # present_score = CH(kmeans, cluster_data)
#         #     # print(str(present_score)+' block index='+str(block_index))
#         #     if present_score > max_score:
#         #         final_kmeans = kmeans
#         #         final_cluster_num = cluster_index_num
#         #         max_score = present_score
#
#         ############
#         kmeans = KMeans(n_clusters=max_cluster_num, max_iter=3000).fit(cluster_data)
#         final_kmeans = kmeans
#         final_cluster_num = max_cluster_num
#     else:
#         final_cluster_num = 0
#     print("block index = "+str(block_index)+", final_cluster_num = " + str(final_cluster_num))
#     # train sgmm
#     train_data = []
#     for i in range(0, final_cluster_num):
#         train_data.append([])
#     if final_cluster_num != 0:
#         for voxel_index in range(0, block_size):
#             count[final_kmeans.labels_[voxel_index]] += 1
#             train_data[final_kmeans.labels_[voxel_index]].append([cluster_data[voxel_index][0], cluster_data[voxel_index][1], cluster_data[voxel_index][2]])
#
#     block = Block()
#     block.cluster_num_ = final_cluster_num
#     for cluster_index in range(0, final_cluster_num):
#         g = mixture.GaussianMixture(n_components=1, tol=1e-5, max_iter=1000)
#         g.fit(train_data[cluster_index])
#         max_bic = g.bic(np.array(train_data[cluster_index]))
#         final_g = g
#         final_component_num = 1
#         max_num = min(ubg, len(train_data[cluster_index]))
#         for component_num in range(2, max_num+1):
#                 g = mixture.GaussianMixture(n_components=component_num, tol=1e-5, max_iter=1000)
#                 g.fit(train_data[cluster_index])
#                 bic_temp = g.bic(np.array(train_data[cluster_index]))
#                 if bic_temp < max_bic:
#                     final_g = g
#                     final_component_num = component_num
#                     max_bic = bic_temp
#
#         # already got final SGMM for cluster_index
#         cluster = Cluster(1.0 * count[cluster_index]/block_size, final_component_num, final_kmeans.cluster_centers_[cluster_index][3])
#         for component_index in range(0, final_component_num):
#             determinant = final_g.covariances_[component_index][0][0] * final_g.covariances_[component_index][1][1] * final_g.covariances_[component_index][2][2]
#             + final_g.covariances_[component_index][0][1] * final_g.covariances_[component_index][1][2] * final_g.covariances_[component_index][2][0]
#             + final_g.covariances_[component_index][0][2] * final_g.covariances_[component_index][1][0] * final_g.covariances_[component_index][2][1]
#             - final_g.covariances_[component_index][0][2] * final_g.covariances_[component_index][1][1] * final_g.covariances_[component_index][2][0]
#             - final_g.covariances_[component_index][0][1] * final_g.covariances_[component_index][1][0] * final_g.covariances_[component_index][2][2]
#             - final_g.covariances_[component_index][0][0] * final_g.covariances_[component_index][1][2] * final_g.covariances_[component_index][2][1]
#             gauss = Gauss(final_g.weights_[component_index], final_g.means_[component_index], final_g.precisions_[component_index], determinant)
#             cluster.add_gauss(gauss)
#         block.add_cluster(cluster)
#
#     return block
#
#
# # train a part of original data
# def train_blocks(result_disk_address,data_source,index, stride, max_cluster_num, src_raw_name, all_data, width, depth, width_num, depth_num, block_size, side,cluster_weights,ubg):
#
#     # print(index,stride,max_cluster_num,src_raw_name,width,depth,width_num,depth_num,block_size,side)
#
#     block_sgmm = [Block()] * stride
#     with open(src_raw_name, 'rb') as f_src:
#         for i in range(0, stride):
#             block_data = read_block(index * stride + i, all_data,width,depth,width_num, depth_num, block_size, side)
#             block_sgmm[i] = train_single_block(index * stride + i, block_data, max_cluster_num,block_size,side,cluster_weights,ubg)
#
#     sgmm_output = result_disk_address + data_source + '_SGMM_Result_Cluster_'+str(index)+'.txt'  # only sgmm arguments
#
#     # restore block_sgmm into txt file
#     with open(sgmm_output, "w") as f_out:
#         for i in range(0, stride):
#             f_out.write(str(block_sgmm[i].cluster_num_)+'\n')
#             for j in range(0, block_sgmm[i].cluster_num_):
#                 f_out.write(str(block_sgmm[i].clusters_[j].probability_)+' '+str(block_sgmm[i].clusters_[j].gauss_count_)+' '+str(block_sgmm[i].clusters_[j].sample_value_)+'\n')
#                 for k in range(0, block_sgmm[i].clusters_[j].gauss_count_):
#                     f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].weight_)+'\n')
#                     f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].determinant_) + '\n')
#                     f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].mean_[0])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].mean_[1])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].mean_[2])+'\n')
#                     f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].precision_[0][0])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].precision_[0][1])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].precision_[0][2])+'\n')
#                     f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].precision_[1][1])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].precision_[1][2])+'\n')
#                     f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].precision_[2][2])+'\n')
#
#     print("training and saving blocks from "+str(index*stride)+" to "+str((index+1)*stride)+" done")


def train_single_block(block_index, block_data, max_cluster_num, block_width, block_depth, block_height, cluster_weights, ubg):
    cluster_data = []
    count = [0] * max_cluster_num[block_index]
    non_zero_count = 0
    for z in range(0, block_height):
        for y in range(0, block_depth):
            for x in range(0, block_width):
                index = z * block_depth * block_width + y * block_width + x
                cluster_data.append([cluster_weights[0]*x, cluster_weights[1]*y, cluster_weights[2]*z, cluster_weights[3] * block_data[index]])
                if block_data[index] != 0:
                    non_zero_count +=1

    block_size = block_width*block_depth*block_height
    if non_zero_count > int(block_size*0.003):
        # print('here3')
        # kmeans = KMeans(n_clusters=2, max_iter=1000).fit(cluster_data)
        # final_kmeans = kmeans
        # final_cluster_num = 2
        # label = kmeans.labels_
        # max_score = silhouette_score(cluster_data, label, metric='euclidean')
        # # max_score = CH(kmeans, cluster_data)
        # for cluster_index_num in range(3, max_cluster_num+1):
        #     print('here4',cluster_index_num)
        #     kmeans = KMeans(n_clusters=cluster_index_num, max_iter=1000).fit(cluster_data)
        #     label = kmeans.labels_
        #     present_score = silhouette_score(cluster_data, label, metric='euclidean')
        #     # present_score = CH(kmeans, cluster_data)
        #     # print(str(present_score)+' block index='+str(block_index))
        #     if present_score > max_score:
        #         final_kmeans = kmeans
        #         final_cluster_num = cluster_index_num
        #         max_score = present_score
        ############

        kmeans = KMeans(n_clusters=max_cluster_num[block_index], max_iter=3000).fit(cluster_data)
        final_kmeans = kmeans
        final_cluster_num = max_cluster_num[block_index]
    else:
        final_cluster_num = 0

    print("block index = "+str(block_index)+", final_cluster_num = " + str(final_cluster_num))
    #train sgmm
    train_data = []
    for i in range(0, final_cluster_num):
        train_data.append([])
    if final_cluster_num != 0:
        for voxel_index in range(0, block_size):
            count[final_kmeans.labels_[voxel_index]] += 1
            train_data[final_kmeans.labels_[voxel_index]].append([cluster_data[voxel_index][0], cluster_data[voxel_index][1], cluster_data[voxel_index][2]])

    block = Block()
    block.cluster_num_ = final_cluster_num
    for cluster_index in range(0, final_cluster_num):
        g = mixture.GaussianMixture(n_components=1, tol=1e-5, max_iter=1000)
        g.fit(train_data[cluster_index])
        max_bic = g.bic(np.array(train_data[cluster_index]))
        final_g = g
        final_component_num = 1
        max_num = min(ubg, len(train_data[cluster_index]))
        for component_num in range(2, max_num+1):
                g = mixture.GaussianMixture(n_components=component_num, tol=1e-5, max_iter=1000)
                g.fit(train_data[cluster_index])
                bic_temp = g.bic(np.array(train_data[cluster_index]))
                if bic_temp < max_bic:
                    final_g = g
                    final_component_num = component_num
                    max_bic = bic_temp

        # already got final SGMM for cluster_index
        cluster = Cluster(1.0 * count[cluster_index]/block_size, final_component_num, final_kmeans.cluster_centers_[cluster_index][3])
        for component_index in range(0, final_component_num):
            determinant = final_g.covariances_[component_index][0][0] * final_g.covariances_[component_index][1][1] * final_g.covariances_[component_index][2][2]
            + final_g.covariances_[component_index][0][1] * final_g.covariances_[component_index][1][2] * final_g.covariances_[component_index][2][0]
            + final_g.covariances_[component_index][0][2] * final_g.covariances_[component_index][1][0] * final_g.covariances_[component_index][2][1]
            - final_g.covariances_[component_index][0][2] * final_g.covariances_[component_index][1][1] * final_g.covariances_[component_index][2][0]
            - final_g.covariances_[component_index][0][1] * final_g.covariances_[component_index][1][0] * final_g.covariances_[component_index][2][2]
            - final_g.covariances_[component_index][0][0] * final_g.covariances_[component_index][1][2] * final_g.covariances_[component_index][2][1]
            gauss = Gauss(final_g.weights_[component_index], final_g.means_[component_index], final_g.precisions_[component_index], determinant)
            cluster.add_gauss(gauss)
        block.add_cluster(cluster)

    return block

debug_info = []

def train_blocks(result_disk_address,
                 data_source,
                 total_num,
                 index,
                 stride,
                 block_info,
                 max_cluster_num,
                 src_raw_name,
                 all_data,
                 width,
                 depth,
                 cluster_weights,
                 ubg):
    number_in_block = total_num - index*stride
    block_sgmm = [Block()] * stride
    # f_debug=open(result_disk_address+data_source+str(index)+".pydbg","w")
    # debug_info=""
    # for i in range(0,stride):
    #     id = index*stride+i
    #     if id >= total_num:
    #         return
    #     xmin = block_info[id][0].x
    #     ymin = block_info[id][0].y
    #     zmin = block_info[id][0].z
    #     xmax = block_info[id][1].x
    #     ymax = block_info[id][1].y
    #     zmax = block_info[id][1].z
    #     debug_info =str(xmin)+" "+str(ymin)+" "+str(zmin)+" "+str(xmax)+" "+str(ymax)+" "+str(zmax)+" "+str(id)+str("\n")
    #     f_debug.write(debug_info)
    # f_debug.close()
    end_block = (index+1)*stride
    end_index = stride
    with open(src_raw_name, 'rb') as f_src:
        for i in range(0, stride):
            if index * stride + i >= total_num:
                end_block = index*stride+i
                end_index = i
                print("the last block breaks at index:" + str(end_block) + ", file no." + str(index))
                break
            block_data = read_block(block_info[index*stride+i][0], block_info[index*stride+i][1], all_data, width, depth)
            block_width = block_info[index*stride+i][1].x - block_info[index*stride+i][0].x
            block_depth = block_info[index*stride+i][1].y - block_info[index*stride+i][0].y
            block_height = block_info[index*stride+i][1].z - block_info[index*stride+i][0].z
            block_sgmm[i] = train_single_block(index*stride+i,
                                               block_data,
                                               max_cluster_num,
                                               block_width=block_width,
                                               block_depth=block_depth,
                                               block_height=block_height,
                                               cluster_weights=cluster_weights,
                                               ubg=ubg)
    sgmm_output = result_disk_address + data_source + '_SGMM_Result_Cluster_'+str(index)+'.txt'  # only sgmm arguments
    with open(sgmm_output, "w") as f_out:
        for i in range(0, end_index):
            f_out.write(str(block_sgmm[i].cluster_num_)+'\n')
            for j in range(0, block_sgmm[i].cluster_num_):
                f_out.write(str(block_sgmm[i].clusters_[j].probability_)+' '+str(block_sgmm[i].clusters_[j].gauss_count_)+' '+str(block_sgmm[i].clusters_[j].sample_value_)+'\n')
                for k in range(0, block_sgmm[i].clusters_[j].gauss_count_):
                    f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].weight_)+'\n')
                    f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].determinant_) + '\n')
                    f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].mean_[0])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].mean_[1])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].mean_[2])+'\n')
                    f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].precision_[0][0])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].precision_[0][1])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].precision_[0][2])+'\n')
                    f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].precision_[1][1])+' '+str(block_sgmm[i].clusters_[j].gausses_[k].precision_[1][2])+'\n')
                    f_out.write(str(block_sgmm[i].clusters_[j].gausses_[k].precision_[2][2])+'\n')
    print("---------------IN FILE:"+str(index)+" training and saving blocks from "+str(index*stride)+" to "+str(end_block)+" done")


def read_block_info_data(path):

    if not os.path.exists(path):
        print('oc file doesn\'t exist')
        exit(0)
    f_oc = open(path)
    num = f_oc.readline()
    lines = f_oc.readlines()

    block_info = []
    for line in lines:
        arr = line.split(' ')
        b = []
        min_point = Point3d(int(arr[0]), int(arr[1]), int(arr[2]))
        max_point = Point3d(int(arr[3]), int(arr[4]), int(arr[5]))
        b.append(min_point)
        b.append(max_point)
        block_info.append(b)
    return block_info

def read_cluster_num(path):
    if not os.path.exists(path):
        print('noc file doesnt exist')
        exit(0)

    f_noc = open(path)
    lines = f_noc.readlines()
    cluster_num = []
    for line in lines:
        cluster_num.append(int(line))
    return cluster_num

# train all block, parallel computing, assign into 4 cpu kernel
if __name__ == '__main__':
    result_disk_address = 'e:/train/'
    process_num = 10  # make sure it is a divisor of block number

    data_source=""
    width = 0
    depth = 0
    height = 0
    process_num = 0

    if len(sys.argv) == 1:
        result_disk_address = raw_input("input disk address:")
        data_source = raw_input('input the data name:')
        width = int(raw_input('weight:'))
        depth = int(raw_input('depth:'))
        height = int(raw_input('height:'))
        process_num = int(raw_input('input the process num (must be the divisor of the block number):'))
    else:
        result_disk_address = sys.argv[1]
        data_source=sys.argv[2]
        width = int(sys.argv[3])
        depth = int(sys.argv[4])
        height = int(sys.argv[5])
        process_num = int(sys.argv[6])

    print("disk address:"+result_disk_address)
    print("data name:"+data_source)
    print("width:"+str(width)+" depth:"+str(depth)+" height:"+str(height))
    print("process num (file num):"+str(process_num))

    src_raw_name = result_disk_address + data_source+'.raw'
    restored_raw_name = result_disk_address+data_source+'_restored.raw'
    result_raw_name = result_disk_address+data_source+'_SGMM_Result.raw'  # restored data
    training_done_signal = result_disk_address + 'Training_Done.txt'

    if not os.path.exists(result_disk_address+data_source+".raw"):
        print('raw file doesn\'t exist')
        exit(0)
    block_info = read_block_info_data(result_disk_address+data_source+'.reoc')

    zero_block_threshold = 0.003
    ubg = 4  # max component number
    cluster_weights = [1.0, 1.0, 1.0, 1.0]  # weights when clustering

    # block_size = side * side * side
    # width_num = width / side
    # depth_num = depth / side
    # height_num = height / side
    # total_num = width_num * depth_num * height_num

    total_num = len(block_info)
    print("block num:"+str(total_num))
    stride = (total_num + process_num - 1) / process_num
    print("stride:" + str(stride))
    restore_raw = bytearray(width * depth * height)

    np.random.seed(1)
    f_all_data = open(src_raw_name, 'rb')
    f_all_data.seek(0, 0)
    all_data = bytearray(f_all_data.read())
    print(len(all_data))
    all_hit = [0] * width * depth * height
    begin_time = time.localtime(time.time())
    cpu_time_begin = time.clock()

    cluster_num = read_cluster_num(result_disk_address + data_source + ".noc")
    for loop_index in [10]:
        print("loop_index = " + str(loop_index))
        proc_record = []
        for i in range(0, process_num):  # a block / 3 seconds
            # Iprint('thread '+str(i))
            # p = Process(target=train_blocks, args=(result_disk_address, data_source, i, stride, loop_index, src_raw_name, all_data, width, depth, width_num, depth_num, block_size, side, cluster_weights,ubg))
            p = Process(target=train_blocks,args=(result_disk_address,
                                            data_source,
                                            total_num,
                                            i,
                                            stride,
                                            block_info,
                                            cluster_num,        #loop_index
                                            src_raw_name,
                                            all_data,
                                            width,
                                            depth,
                                            cluster_weights,
                                            ubg))
            p.start()
            proc_record.append(p)

        for p in proc_record:
            p.join()
        print("training SGMM done.")
        cpu_time_end = time.clock()
        print time.strftime('Training begin at %Y-%m-%d %H:%M:%S', begin_time)
        print time.strftime('Training done at %Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("cpu time cost in python:" + str(cpu_time_end - cpu_time_begin)+".s")

        # single_block_data = read_block(5764)
        # train_single_block(5764, single_block_data, loop_index)
