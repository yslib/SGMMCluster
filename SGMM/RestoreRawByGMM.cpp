#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#include "funcs.h"
#define M_PI 3.14159265358979323846

//
//static struct Gauss1 {
//	float weight_ = 0.0;
//	float mean_ = 0.0;
//	float covariance_ = 0.0;
//};
//static struct gmmBlock {
//	unsigned char gauss_num_;
//	Gauss1 gausses_[12];
//};


static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}


//
__global__
void BlockGMMRestoreVoxel(unsigned char * raw_result, 
	int width, 
	int depth, 
	int height, 
	int side,
	int width_num, 
	int depth_num, 
	int height_num,
	gmmBlock* block_data_device,
	double* temp_p,
	double* temp_p2) {
	//
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //voxel index
	if (calc_index >= width*depth*height)
		return;
	
	//voxel global coordinate
	int global_height_pos = calc_index / (depth * width);
	int global_depth_pos = (calc_index - global_height_pos * depth * width) / width;
	int global_width_pos = calc_index - global_height_pos * depth * width - global_depth_pos * width;

	//block coordinate
	int height_index = global_height_pos / side;
	int depth_index = global_depth_pos / side;
	int width_index = global_width_pos / side;

	//block index
	int block_index = height_index * (width_num * depth_num) + depth_index * width_num + width_index;

	//
	int sample_value = 125;
	curandState state;
	curand_init(calc_index, 0, 0, &state);
	//printf("block index:%d gauss count:%d\n", block_index,
	//	block_data_device[block_index].gauss_num_);
	while (true) {
		double sample_uniform = curand_uniform(&state);
		int component_selected = 0;
		double total_value = 0.0;
		for (int gauss_index = 0; gauss_index < block_data_device[block_index].gauss_num_; gauss_index++) {
			total_value += block_data_device[block_index].gausses_[gauss_index].weight_;
			if (sample_uniform < total_value) {
				component_selected = gauss_index;
				break;
			}
		}
		sample_value = block_data_device[block_index].gausses_[component_selected].covariance_ * curand_normal(&state) + block_data_device[block_index].gausses_[component_selected].mean_;
		//printf("%f\n", curand_normal(&state));
		//printf("%d %f %f\n", block_index, block_data_device[block_index].gausses_[component_selected].covariance_, block_data_device[block_index].gausses_[component_selected].mean_);
		if (sample_value >= 0.0 && sample_value <= 255.0) {
			break;
		}
	}
	//printf("%d %d\n",block_index, sample_value);
	raw_result[calc_index] = sample_value;
}


int restore_raw_by_gmm(int argc, char **argv)
{
	const int blockSize = 32;

	int width;
	int depth;
	int height;
	int side;
	std::string data_source;
	std::string disk_address;
	std::string src_address;
	std::string result_name;
	std::string gmm_binary_address;
	int width_num;
	int depth_num;
	int height_num;
	int block_num;
	int block_size;
	int total_size;
	unsigned char * raw_result;
	unsigned char * raw_src;
	double * temp_p;
	double * temp_p2;

	gmmBlock* block_data_device;
	gmmBlock* block_data_host;

	std::cout << "----------------RESTORE RAW BY BLOCK GMM MODULE---------------\n";
	std::cout << "input data address\n";
	std::cin >> disk_address;

	std::cout << "input data name\n";
	std::cin >> data_source;

	src_address = disk_address + data_source + ".raw";
	gmm_binary_address = disk_address + data_source + "_GMM_Result.gmm";
	result_name = disk_address + data_source + "_restored_gmm.raw";
	std::cout << "input width depth height side (4)\n";
	std::cin >> width >> depth >> height >> side;

	assert(width%side == 0);
	width_num = width / side;
	assert(depth%side == 0);
	depth_num = depth / side;
	assert(height%side == 0);
	height_num = height / side;

	block_num = width_num * depth_num * height_num;
	block_size = side * side *side;
	total_size = width * depth * height;

	std::cout << "width:" << width << std::endl;
	std::cout << "depth:" << depth << std::endl;
	std::cout << "height:" << height << std::endl;
	std::cout << "block num:" << block_num << std::endl;
	std::cout << "block size:" << block_size << std::endl;
	std::cout << "total size:" << total_size << std::endl;

	std::cout << "Part0: Initializing..." << std::endl;

	unsigned char * raw_result_host = (unsigned char *)malloc(total_size * sizeof(unsigned char));
	unsigned char * temp_p_host = (unsigned char *)malloc(total_size * sizeof(double));
	unsigned char * temp_p2_host = (unsigned char *)malloc(total_size * sizeof(double));

	//cudaMalloc(&raw_src, total_size * sizeof(unsigned char));
	CUDA_CALL(cudaMalloc(&raw_result, total_size * sizeof(unsigned char)));
	CUDA_CALL(cudaMalloc(&temp_p, total_size * sizeof(double)));
	CUDA_CALL(cudaMalloc(&temp_p2, total_size * sizeof(double)));

	for (int i = 0; i < total_size; i++) {
		raw_result_host[i] = 125;
		temp_p_host[i] = 0.000;
		temp_p2_host[i] = 0.000;
	}
	CUDA_CALL(cudaMemcpy(raw_result, raw_result_host, total_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(temp_p, temp_p_host, total_size * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(temp_p2, temp_p2_host, total_size * sizeof(double), cudaMemcpyHostToDevice));

	// Part1:
	std::cout << std::endl << "Part1: Reading GMMs..." << std::endl;
	cudaMalloc(&block_data_device, block_num * sizeof(gmmBlock));
	block_data_host = new gmmBlock[block_num];
	std::ifstream f_gmm(gmm_binary_address, std::ios::binary);
	if (f_gmm.is_open() == false) {
		std::cout << "can not open file\n";
		return 0;
	}
	for (int block_index = 0; block_index < block_num; block_index++) {
		f_gmm.read((char*)&(block_data_host[block_index].gauss_num_), sizeof(unsigned char));
		for (int gauss_index = 0; gauss_index < block_data_host[block_index].gauss_num_; gauss_index++) {
			f_gmm.read((char*)&(block_data_host[block_index].gausses_[gauss_index].weight_), sizeof(float));
			f_gmm.read((char*)&(block_data_host[block_index].gausses_[gauss_index].mean_), sizeof(float));
			f_gmm.read((char*)&(block_data_host[block_index].gausses_[gauss_index].covariance_), sizeof(float));
		}
	}
	f_gmm.close();
	CUDA_CALL(cudaMemcpy(block_data_device, block_data_host, block_num * sizeof(gmmBlock), cudaMemcpyHostToDevice));


	// Part2:
	std::cout << std::endl << "Part2: Restoring data..." << std::endl;
	int numBlocks = (total_size + blockSize - 1) / blockSize;
	int start = clock();
	BlockGMMRestoreVoxel <<<numBlocks, blockSize >> > (raw_result, width, depth, height, side, width_num, depth_num, height_num, block_data_device,
		temp_p, temp_p2);
	CUDA_CALL(cudaDeviceSynchronize());
	int finish = clock();
	std::cout << "Restoring time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
	// Part3:
	std::cout << std::endl << "Part3: Saving raw file..." << std::endl;
	std::ofstream f_result(result_name, std::ios::binary);
	CUDA_CALL(cudaMemcpy(raw_result_host, raw_result, total_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	f_result.write((const char*)raw_result_host, total_size);
	f_result.close();

	std::cout << "Creating .vifo file\n";
	create_vifo_file(disk_address, data_source + "_restored_gmm", width, depth, height);

	cudaFree(raw_result);
	cudaFree(temp_p);
	cudaFree(temp_p2);
	std::cin.get();
	return 0;
}