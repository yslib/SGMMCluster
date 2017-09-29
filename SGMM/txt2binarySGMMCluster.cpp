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

// ������������ begin
#define MAX_CLUSTER_NUM 14
#define MAX_GAUSS_COMPONENT_NUM 4

// �������ݽṹ
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
	unsigned char cluster_num; // ʵ�ʾ��������
	Cluster clusters_[MAX_CLUSTER_NUM];
	sgmmClusterBlock() {
		cluster_num = 0;
	}
};

// ����ת�ַ���
static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}

int txt2binarySGMMCluster(int argc,char ** argv) {

	//constant varible
	static const int n = 3; // ����ά��
	std::cout << "---------- TXT TO BINARY SGMM CLUSTER MODULE-------------\n";

	static std::string data_source; // ����������ݵ�����
	static std::string disk_address;
	static int width; // �����
	static int depth; // ������
	static int height;  // �����
	static int sgmm_file_num;

	static std::string result_name;
	static int block_num;
	static int stride;
	static int total_size;
	static unsigned char * raw_src;
	static unsigned char * raw_result;
	static double * temp_p; // ������1
	static double * temp_p2; // ������2
	static int * zero_count;
	static int * calc_count;

	// Part1: ��ȡSGMM����
	std::cout << "input data address\n";
	std::cin >> disk_address;
	std::cout << "input the data name\n";
	std::cin >> data_source;
	std::cout << "input width depth height file_num\n";
	std::cin >> width >> depth >> height >> sgmm_file_num;

	//��ȡ�������
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
	for (int i = 0; i < sgmm_file_num; i++) {
		std::string sgmm_input_name = disk_address + data_source + "_SGMM_Result_Cluster_" + Int2String(i) + ".txt";
		std::cout << "Reading file " << sgmm_input_name << "..." << std::endl;
		std::ifstream sgmm_input(sgmm_input_name);
		if (!sgmm_input.is_open())
		{
			std::cout << "Reading SGMM file failed.";
			exit(1);
		}

		//����stride����
		for (int j = 0; j < stride; j++) {
			int cluster_num; //���ʵ�ʾ�������
			sgmm_input >> cluster_num;
			assert(cluster_num <= 10);

			sgmmClusterBlock single_block;
			single_block.cluster_num = cluster_num;

			//����cluster_num��cluster
			for (int k = 0; k < cluster_num; k++) {
				double probability;
				int gauss_count;
				double sample_value;
				sgmm_input >> probability >> gauss_count >> sample_value;

				Cluster single_cluster;
				single_cluster.probability_ = probability;
				single_cluster.gauss_count_ = gauss_count;
				single_cluster.sample_value_ = sample_value;
				//����gauss_num����˹��֧
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
	std::cout << "block in total:" << count << std::endl;
	// Part2: �洢SGMMΪ������
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