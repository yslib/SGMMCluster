#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cassert>

struct sgmmGauss {
	float weight_ = 0.0;
	float mean_[3] = { 0.0,0.0,0.0 };
	float covariance_[6] = { 0.0,0.0,0.0,0.0,0.0,0.0 };
};

struct Bin {
	float probability_ = 0.0;
	unsigned char gauss_count_ = 0;
	sgmmGauss gausses_[12];
};

struct sgmmBlock {
	int gauss_count_ = 0; // test only
	unsigned char bin_num_ = 0;
	unsigned char bin_indexs_[128];
	Bin bins_[128]; // 
};

static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}

int txt2binarysgmm(int argc,char ** argv)
{
	//constant varible
	const int max_bin_num = 128; // 
	const int max_sgmm_component_num = 4; // 
	const int n = 3; //
	const int blockSize = 256; // 
	const int integration_scale = 3; // 
	std::cout << "-------------TXT TO BINARY SGMM MODULE ---------------\n";

	std::string data_source;
	std::string disk_address;
	int width;
	int depth;
	int height;
	int side;
	int sgmm_file_num;

	std::string sgmm_binary_address;
	int width_num;
	int depth_num;
	int height_num;
	int block_num;
	int stride;
	int block_size;
	int total_size;
	unsigned char * raw_src;
	unsigned char * raw_result;
	double * temp_p; // 
	double * temp_p2; // 
	int * zero_count;
	int * calc_count;


	// Part1:
	std::cout << "input data address\n";
	std::cin >> disk_address;

	std::cout << "input the data name\n";
	std::cin >> data_source;

	std::cout << "input width depth height file_num side\n";
	std::cin >> width >> depth >> height >> sgmm_file_num >> side;

	sgmm_binary_address = disk_address + data_source + "_SGMM_Result.sgmm";

	assert(width%side == 0);
	width_num = width / side;
	assert(depth%side == 0);
	depth_num = depth / side;
	assert(height%side == 0);
	height_num = height / side;

	block_num = width_num * depth_num * height_num;
	std::cout << "block number:" << block_num << std::endl;
	stride = (block_num+sgmm_file_num-1) / sgmm_file_num;
	std::cout << "stride:" << stride << std::endl;
	block_size = side * side *side;
	std::cout << "block size:" << block_size << std::endl;
	total_size = width * depth * height;
	std::cout << "total size:" << total_size << std::endl;


	sgmmBlock* block_data = new sgmmBlock[block_num];
	std::cout << std::endl << "Part1: Reading SGMMs..." << std::endl;
	int count = 0;
	//reading training text file
	for (int i = 0; i < sgmm_file_num; i++) {
		std::string sgmm_input_name = disk_address + data_source + "_SGMM_Result_" + Int2String(i) + ".txt";
		std::cout << "Reading file " << sgmm_input_name << "..." << std::endl;
		std::ifstream sgmm_input(sgmm_input_name);
		if (!sgmm_input.is_open())
		{
			std::cout << "Reading SGMM file failed.";
			exit(1);
		}

		//
		for (int j = 0; j < stride; j++) {
			int block_index = i * stride + j;		//absolute index for all files
			int bin_num = 0;

			sgmm_input >> bin_num;
			//
			assert(bin_num >= 0);
			assert(bin_num <= max_bin_num);
			
			block_data[block_index].bin_num_ = bin_num;
			if (block_index == 0) {
				std::cout << bin_num << std::endl;
			}
			for (int k = 0; k < bin_num; k++) {
				if (block_index == 0) {
					std::cout << "-----------------GAUSS :" << k << std::endl;
				}
				int real_index = 0;
				float probability = 0.0;
				int gauss_count = 0;
				sgmm_input >> real_index >> probability >> gauss_count;
				assert(real_index >= 0);
				assert(real_index < max_bin_num);
				assert(gauss_count >= 0);
				assert(gauss_count <= max_sgmm_component_num);
				block_data[block_index].bin_indexs_[k] = real_index;
				block_data[block_index].bins_[real_index].probability_ = probability;
				block_data[block_index].bins_[real_index].gauss_count_ = gauss_count;
				if (block_index == 0) {
					std::cout << "bin index:" << real_index << std::endl;
					std::cout << "probability:" << probability << std::endl;
					std::cout << "gauss count:" << gauss_count << std::endl;
				}
				//read information for every gaussian component
				for (int p = 0; p < gauss_count; p++) {

					block_data[block_index].gauss_count_++;  // test only
					float weight = 0.0;
					float mean[3] = { 0.0 ,0.0, 0.0 };
					float covariance[6] = { 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0 }; //symetric covariance matrix 
					sgmm_input >> weight;
					sgmm_input >> mean[0] >> mean[1] >> mean[2];
					sgmm_input >> covariance[0] >> covariance[1] >> covariance[2] >> covariance[3] >> covariance[4] >> covariance[5];
					if (block_index == 0) {
						std::cout << "GAUSS COUNT:" << p << std::endl;
						std::cout <<"weight:"<< weight << std::endl;
						std::cout <<"mean:"<< mean[0] << " " << mean[1] << " " << mean[2] << std::endl;
						std::cout << "covariance:"<<covariance[0] << " " << covariance[1] << " " << covariance[2] << " " << covariance[3] <<
							covariance[4] << " " << covariance[5] << std::endl;
					}
					block_data[block_index].bins_[real_index].gausses_[p].weight_ = weight;
					for (int loop = 0; loop < 3; loop++) {
						block_data[block_index].bins_[real_index].gausses_[p].mean_[loop] = mean[loop];
					}
					for (int loop = 0; loop < 6; loop++) {
						block_data[block_index].bins_[real_index].gausses_[p].covariance_[loop] = covariance[loop];
					}
				}
			}
			count++;
			if (count == block_num) {
				break;
			}
		}

		sgmm_input.close();
		//remove(sgmm_input_name.c_str());
	}
	std::cout << "block in total:" << count << std::endl;

	// Test only
	int total_count = 0;
	for (int i = 0; i < block_num; i++) {
		total_count += block_data[i].gauss_count_;
	}
	std::cout << "average gauss count for every block = " << total_count / block_num << std::endl;

	// Part2: writing as binary file
	std::ofstream f_sgmm(sgmm_binary_address, std::ios::binary);
	for (int block_index = 0; block_index < block_num; block_index++) {
		f_sgmm.write((char*)&(block_data[block_index].bin_num_), sizeof(unsigned char));
		for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
			int real_bin_index = block_data[block_index].bin_indexs_[bin_count];
			f_sgmm.write((char*)&(real_bin_index), sizeof(unsigned char));
			f_sgmm.write((char*)&(block_data[block_index].bins_[real_bin_index].probability_), sizeof(float));
			f_sgmm.write((char*)&(block_data[block_index].bins_[real_bin_index].gauss_count_), sizeof(unsigned char));
			for (int gauss_index = 0; gauss_index < block_data[block_index].bins_[real_bin_index].gauss_count_; gauss_index++) {
				f_sgmm.write((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].weight_), sizeof(float));
				for (int i = 0; i < 3; i++) {
					f_sgmm.write((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].mean_[i]), sizeof(float));
				}
				for (int i = 0; i < 6; i++) {
					f_sgmm.write((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[i]), sizeof(float));
				}
			}
		}
	}
	f_sgmm.close();
	std::cin.get();
	return 0;
}