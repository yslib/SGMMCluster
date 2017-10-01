#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cassert>


struct Gauss1 {
	float weight_ = 0.0;
	float mean_ = 0.0;
	float covariance_ = 0.0;
};


struct gmmBlock {
	unsigned char gauss_num_ = 12;
	Gauss1 gausses_[12];
};


static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	return s_temp.str();
}

int txt2binarygmm(int argc,char ** argv)
{
	std::cout << "----------TXT TO BINART GMM MODULE\n";
	int width;
	int depth;
	int height;
	int side;
	int sgmm_file_num;
	std::string data_source;
	std::string disk_address;
	std::string gmm_binary_address;
	int width_num;
	int depth_num;
	int height_num;
	int block_num;
	int stride;

	// Part1:
	std::cout << "input data address\n";
	std::cin >> disk_address;
	std::cout << "input the data name\n";
	std::cin >> data_source;
	std::cout << "input width depth height file_num side\n";
	std::cin >> width >> depth >> height >> sgmm_file_num >> side;

	gmm_binary_address = disk_address + data_source + "_GMM_Result.gmm";
	assert(width%side == 0);
	width_num = width / side;
	assert(depth%side == 0);
	depth_num = depth / side;
	assert(height%side == 0);
	height_num = height / side;

	block_num = width_num * depth_num * height_num;

	stride = (block_num+sgmm_file_num-1) / sgmm_file_num;
	std::cout << "stride:" << stride << std::endl;
	std::cout << disk_address << std::endl;
	std::cout << data_source << std::endl;
	std::cout << width << " " << depth << " " << height << " " << sgmm_file_num << " " << side << std::endl;

	gmmBlock* block_data = new gmmBlock[block_num];
	int total_gauss_count = 0;
	int count = 0;
	for (int i = 0; i < sgmm_file_num; i++) {
		std::string idx = Int2String(i);
		std::string gmm_input_name = disk_address + data_source + "_GMM_Result_" + idx + ".txt";
		//std::string gmm_input_name = "e:/train/test/test_GMM_Result_0.txt";
		std::cout << "Reading file " << gmm_input_name << "..." << std::endl;

		std::ifstream gmm_input(gmm_input_name);

		if (!gmm_input.is_open())
		{
			std::cout << "Reading GMM file failed.";
			exit(1);
		}
		for (int block_index = 0; block_index < stride; block_index++) {
			int final_index = i * stride + block_index;
			int temp_int;
			gmm_input >> temp_int;
			block_data[final_index].gauss_num_ = temp_int;
			total_gauss_count += temp_int;
			for (int gauss_index = 0; gauss_index < block_data[final_index].gauss_num_; gauss_index++) {
				gmm_input >> block_data[final_index].gausses_[gauss_index].weight_;
				gmm_input >> block_data[final_index].gausses_[gauss_index].mean_;
				gmm_input >> block_data[final_index].gausses_[gauss_index].covariance_;
			}
			count++;
			if (count == block_num) {
				break;
			}
		}
		gmm_input.close();
		//remove(gmm_input_name.c_str());
	}
	if (count == 0) {
		std::cout << "No Block\n";
		return 0;
	}
	std::cout << "block in total:" << count << std::endl;
	std::cout << "gauss in total:" << total_gauss_count << std::endl;
	std::cout << "average gauss per block:" << total_gauss_count / count << std::endl;
	// Part2:
	std::ofstream f_gmm(gmm_binary_address, std::ios::binary);
	for (int block_index = 0; block_index < block_num; block_index++) {
		f_gmm.write((char*)&(block_data[block_index].gauss_num_), sizeof(unsigned char));
		for (int gauss_index = 0; gauss_index < block_data[block_index].gauss_num_; gauss_index++) {
			f_gmm.write((char*)&(block_data[block_index].gausses_[gauss_index].weight_), sizeof(float));
			f_gmm.write((char*)&(block_data[block_index].gausses_[gauss_index].mean_), sizeof(float));
			f_gmm.write((char*)&(block_data[block_index].gausses_[gauss_index].covariance_), sizeof(float));
		}
	}
	f_gmm.close();
	std::cin.get();
	return 0;
}