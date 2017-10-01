#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cassert>

// 参数控制区域 begin
#define MAX_CLUSTER_NUM 14
#define MAX_GAUSS_COMPONENT_NUM 4

// 基本数据结构
struct sgmmClusterGauss {
	float weight_;
	float mean_[3];
	float precisions_[9];
	float determinant_;
	sgmmClusterGauss() :weight_{ 0.0 }, determinant_{ 1.0 } {
		for (int i = 0; i < 3; i++) {
			mean_[i] = 0.0;
		}
		for (int i = 0; i < 9; i++) {
			precisions_[i] = 0.0;
		}
	}
};

struct Cluster {
	float probability_ = 0.0;
	unsigned char gauss_count_ = 0;
	float sample_value_ = 0;
	sgmmClusterGauss gausses_[MAX_GAUSS_COMPONENT_NUM];
};

struct sgmmClusterBlock {
	unsigned char cluster_num; // 实际聚类的数量
	Cluster clusters_[MAX_CLUSTER_NUM];
	sgmmClusterBlock() {
		cluster_num = 0;
	}
};

// 整数转字符串
static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}

int txt2binarySGMMCluster(int argc,char ** argv) {

	//constant varible
	const int n = 3; // 数据维度
	std::cout << "---------- TXT TO BINARY SGMM CLUSTER MODULE-------------\n";

	std::string data_source; // 被处理的数据的名字
	std::string disk_address;
	int width; // 物体宽
	int depth; // 物体深
	int height;  // 物体高
	int sgmm_file_num;

	std::string result_name;
	int block_num;
	int stride;
	int total_size;
	unsigned char * raw_src;
	unsigned char * raw_result;
	double * temp_p; // 调试用1
	double * temp_p2; // 调试用2
	int * zero_count;
	int * calc_count;

	// Part1: 读取SGMM数据
	std::cout << "input data address\n";
	std::cin >> disk_address;
	std::cout << "input the data name\n";
	std::cin >> data_source;
	std::cout << "input width depth height file_num\n";
	std::cin >> width >> depth >> height >> sgmm_file_num;

	//读取块的数量
	std::fstream bounding_box_file(disk_address + data_source + std::string{ ".reoc" });
	if (bounding_box_file.is_open() == false) {
		std::cout << "can not open bounding box file\n";
		return 0;
	}
	bounding_box_file >> block_num;
	std::cout << "block num:" << block_num << std::endl;
	bounding_box_file.close();


	//stride = block_num / sgmm_file_num;
	stride = (block_num + sgmm_file_num - 1) / sgmm_file_num;
	std::cout << "stride:" << stride << std::endl;
	total_size = width * depth * height;
	std::cout << "total size:" << total_size << std::endl;
	std::string sgmm_binary_address = disk_address + data_source + "_SGMM_Cluster_Result.sgmm";
	sgmmClusterBlock* block_data = new sgmmClusterBlock[block_num];


	std::cout << std::endl << "Part1: Reading SGMMs..." << std::endl;

	int count = 0;
	int total_cluster_count = 0;
	int total_gauss_count = 0;
	for (int i = 0; i < sgmm_file_num; i++) {
		std::string sgmm_input_name = disk_address + data_source + "_SGMM_Result_Cluster_" + Int2String(i) + ".txt";
		std::cout << "Reading file " << sgmm_input_name << "..." << std::endl;
		std::ifstream sgmm_input(sgmm_input_name);
		if (!sgmm_input.is_open())
		{
			std::cout << "Reading SGMM file failed.";
			exit(1);
		}

		//遍历stride个块
		for (int j = 0; j < stride; j++) {
			int cluster_num; 
			sgmm_input >> cluster_num;
			total_cluster_count += cluster_num;
			assert(cluster_num <= 10);

			sgmmClusterBlock single_block;
			single_block.cluster_num = cluster_num;

			//遍历cluster_num个cluster
			for (int k = 0; k < cluster_num; k++) {
				double probability;
				int gauss_count;
				double sample_value;
				sgmm_input >> probability >> gauss_count >> sample_value;
				total_gauss_count += gauss_count;
				Cluster single_cluster;
				single_cluster.probability_ = probability;
				single_cluster.gauss_count_ = gauss_count;
				single_cluster.sample_value_ = sample_value;
				//遍历gauss_num个高斯分支
				for (int p = 0; p < gauss_count; p++) {
					double weight;
					double determinant;
					double mean[3];
					double precision[6];
					sgmm_input >> weight;
					sgmm_input >> determinant;
					sgmm_input >> mean[0] >> mean[1] >> mean[2];
					sgmm_input >> precision[0] >> precision[1] >> precision[2] >> precision[3] >> precision[4] >> precision[5];
					sgmmClusterGauss single_gauss;

					single_gauss.weight_ = weight;
					single_gauss.determinant_ = determinant;

					for (int loop = 0; loop < 3; loop++) {
						single_gauss.mean_[loop] = mean[loop];
					}
					for (int loop = 0; loop < 6; loop++) {
						single_gauss.precisions_[loop] = precision[loop];
					}

					single_cluster.gausses_[p] = single_gauss;

				}
				single_block.clusters_[k] = single_cluster;

			}

			block_data[i*stride + j] = single_block;
			count++;
			if (count == block_num) {
				break;
			}
		}

		sgmm_input.close();
		//remove(sgmm_input_name.c_str());
	}
	if (count == 0) {
		std::cout << "No Block\n";
		return 0;
	}
	//
	std::cout << "block in total:" << count << std::endl;
	std::cout << "Gauss count:" << total_gauss_count << std::endl;
	std::cout << "Cluster Count:" << total_cluster_count << std::endl;
	std::cout << "average gauss count per block:" << total_gauss_count / count << std::endl;
	std::cout << "average cluster count per block:" << total_cluster_count / count << std::endl;
	if(total_cluster_count != 0) std::cout << "average gauss count per cluster:" << total_gauss_count / total_cluster_count << std::endl;

	// Part2: Save as binary file
	std::ofstream f_sgmm(sgmm_binary_address, std::ios::binary);
	for (int block_index = 0; block_index < block_num; block_index++) {
		//std::cout << block_index << std::endl;
		assert(block_data[block_index].cluster_num <= 10);
		f_sgmm.write((char*)&(block_data[block_index].cluster_num), sizeof(unsigned char));
		for (int cluster_index = 0; cluster_index < block_data[block_index].cluster_num; cluster_index++) {
			f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].probability_), sizeof(float));
			f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].gauss_count_), sizeof(unsigned char));
			f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].sample_value_), sizeof(float));
			for (int gauss_index = 0; gauss_index < block_data[block_index].clusters_[cluster_index].gauss_count_; gauss_index++) {
				f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].weight_), sizeof(float));
				f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].determinant_), sizeof(float));
				for (int i = 0; i < 3; i++) {
					f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].mean_[i]), sizeof(float));
				}
				for (int i = 0; i < 6; i++) {
					f_sgmm.write((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[i]), sizeof(float));
				}
			}
		}
	}
	f_sgmm.close();


	std::cin.get();
	return 0;
}