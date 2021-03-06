﻿#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "funcs.h"
#define M_PI 3.14159265358979323846

#define DISPLAY_CUDA_CALL_ERROR_INFO

#ifdef DISPLAY_CUDA_CALL_ERROR_INFO

#define CUDA_CALL(x) {const cudaError_t a =(x);\
if(a != cudaSuccess){printf("\nCUdaError:%s(line:%d)\n",\
cudaGetErrorString(a),__LINE__);cudaDeviceReset();}}

#else
#define CUDA_CALL(x) (x)
#endif

//#define BIN_NUM_EXISTS_CHECK





struct sgmmGauss {
	float weight_ = 0.0;
	float mean_[3] = { 0.0,0.0,0.0 };
	float covariance_[9] = { 0.0,0.0,0.0 ,0.0,0.0,0.0 ,0.0, 0.0 ,0.0 };
};

struct Bin {
	float probability_ = 0.0;
	unsigned char gauss_count_ = 0;
	sgmmGauss gausses_[12];
};

struct sgmmBlock {
	unsigned char bin_num_ = 0;
	unsigned char bin_indexs_[128];
	Bin bins_[128];
};

struct sgmmIntegrations {
	float integration_numerator[128];
	float integration_denominator[128];
	float integration_value[128];
};
static sgmmIntegrations* all_block_integrations;



//
__host__ __device__
static double MulMatrix(double * A, double * B, double * C) {
	return (A[0] * B[0] + A[1] * B[3] + A[2] * B[6]) * C[0] + (A[0] * B[1] + A[1] * B[4] + A[2] * B[7]) * C[1] + (A[0] * B[2] + A[1] * B[5] + A[2] * B[8]) * C[2];
}


__device__
static inline double CalcGMM(double* mean, double* cov, double* local_pos, int n) {
	double determinant = cov[0] * cov[4] * cov[8] + cov[1] * cov[5] * cov[6] + cov[2] * cov[3] * cov[7] //Ð­·½²î¾ØÕóµÄÐÐÁÐÊ½
		- cov[2] * cov[4] * cov[6] - cov[1] * cov[3] * cov[8] - cov[0] * cov[5] * cov[7];

	double inv_cov[9]; //
	inv_cov[0] = (cov[4] * cov[8] - cov[7] * cov[5]) / determinant;
	inv_cov[1] = -(cov[1] * cov[8] - cov[7] * cov[2]) / determinant;
	inv_cov[2] = (cov[1] * cov[5] - cov[2] * cov[4]) / determinant;
	inv_cov[3] = (cov[5] * cov[6] - cov[8] * cov[3]) / determinant;
	inv_cov[4] = -(cov[2] * cov[6] - cov[8] * cov[0]) / determinant;
	inv_cov[5] = (cov[2] * cov[3] - cov[0] * cov[5]) / determinant;
	inv_cov[6] = (cov[3] * cov[7] - cov[6] * cov[4]) / determinant;
	inv_cov[7] = -(cov[0] * cov[7] - cov[6] * cov[1]) / determinant;
	inv_cov[8] = (cov[0] * cov[4] - cov[1] * cov[3]) / determinant;

	double diff[3]; //
	for (int i = 0; i < 3; i++) {
		diff[i] = local_pos[i] - mean[i];
	}

	double left = 1.0 / (sqrt(8 * M_PI * M_PI * M_PI) * sqrt(determinant));
	double matrix_result = MulMatrix(diff, inv_cov, diff);
	//printf("%lf %lf\n", left, matrix_result);
	double right = __expf(-0.5 * matrix_result);
	//printf("%lf", right);
	return left * right;
	//return 0;
}


// ¼ÆËãlocal_posÎ»ÖÃ´¦µÄsgmm£¨¸ÅÂÊÃÜ¶È£©
__device__
static double CalcSGMM(double* local_pos, int gauss_count, sgmmBlock* block_data, int block_index, int bin_index, int n) {
	double sgmm_result = 0.0;
	for (int gauss_index = 0; gauss_index < gauss_count; gauss_index++) {
		double mean[3];	//
		for (int i = 0; i < 3; i++) {
			mean[i] = block_data[block_index].bins_[bin_index].gausses_[gauss_index].mean_[i];
		}

		double cov[9]; //
		for (int i = 0; i < 9; i++) {
			cov[i] = block_data[block_index].bins_[bin_index].gausses_[gauss_index].covariance_[i];
		}

		double added_value = block_data[block_index].bins_[bin_index].gausses_[gauss_index].weight_*CalcGMM(mean, cov, local_pos, n);

		sgmm_result += added_value; //local_pos
	}
	return sgmm_result;
}


//
__global__
void CalcIntegrationsNumerator(sgmmIntegrations* all_block_integrations, sgmmBlock* block_data, int side, int n, int block_num, int bin_num, double * temp_p, double * temp_p2) {
	
	double offset = 0.000;
	//bin index
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; 

	if (calc_index >= block_num * bin_num) return;
	//printf("%d\n", calc_index);
	int block_index = calc_index / bin_num;
	int bin_index = calc_index - block_index * bin_num;
	// Test code begin
	//if (block_index == 2500 && bin_index == 0) {
	//	all_block_integrations[block_index].integration_numerator[bin_index] = 7.77777777;
	//}
	// Test code end


	bool exist = false;
	for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
		if (block_data[block_index].bin_indexs_[bin_count] == bin_index) {
			exist = true;
			break;
		}
	}
	if (!exist) {
		return;
	}


	//if the bin has no gaussian component,the integration will be 0
	if (block_data[block_index].bins_[bin_index].gauss_count_ == 0) {
		all_block_integrations[block_index].integration_numerator[bin_index] = 0.0;
		return;
	}

	double numerator = 0.0;
	int gauss_count = block_data[block_index].bins_[bin_index].gauss_count_;
	for (int z = 0; z < side; z++) {
		for (int y = 0; y < side; y++) {
			for (int x = 0; x < side; x++) {
				double local_pos[3] = { x + offset,y + offset,z + offset };
				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, bin_index, n);
				numerator += sgmm;
			}
		}
	}

	all_block_integrations[block_index].integration_numerator[bin_index] = numerator;

	//printf("%f\n", numerator);
}

__global__
void CalcIntegrationsDenominator(sgmmIntegrations* all_block_integrations, sgmmBlock* block_data, int side, int n, int block_num, int bin_num, int loop_index, int integration_scale, double * temp_p, double * temp_p2) {


	double offset = 0.000;
	//bin index
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (calc_index >= block_num * bin_num) return;//out of range
	int block_index = calc_index / bin_num;
	int bin_index = calc_index - block_index * bin_num;


	bool exist = false;
	for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
		if (block_data[block_index].bin_indexs_[bin_count] == bin_index) {
			exist = true;
			break;
		}
	}
	if (!exist) {
		return;
	}



	if (block_data[block_index].bins_[bin_index].gauss_count_ == 0) {
		all_block_integrations[block_index].integration_denominator[bin_index] += 0.0;
		if (loop_index == integration_scale*integration_scale*integration_scale - 1) {
			if (all_block_integrations[block_index].integration_denominator[bin_index] == 0) {
				all_block_integrations[block_index].integration_value[bin_index] = 0.0;
			}
			else {
				all_block_integrations[block_index].integration_value[bin_index] = all_block_integrations[block_index].integration_numerator[bin_index] / all_block_integrations[block_index].integration_denominator[bin_index];
				
			}
		}
		return;
	}


	int z_index = loop_index / (integration_scale*integration_scale);
	int y_index = (loop_index - z_index * (integration_scale*integration_scale)) / integration_scale;
	int x_index = loop_index - z_index * (integration_scale*integration_scale) - y_index * integration_scale;

	double denominator = 0.0;
	int gauss_count = block_data[block_index].bins_[bin_index].gauss_count_;
	for (int z = (-1 + z_index) * side; z < (0 + z_index) * side; z++) {
		for (int y = (-1 + y_index) * side; y < (0 + y_index) * side; y++) {
			for (int x = (-1 + x_index) * side; x < (0 + x_index) * side; x++) {
				double local_pos[3] = { x + offset,y + offset,z + offset };
				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, bin_index, n);
				denominator += sgmm;
			}
		}
	}
	all_block_integrations[block_index].integration_denominator[bin_index] += denominator;
	//printf("d: %f\n", denominator);
	//printf("inter result:%f\n", all_block_integrations[block_index].integration_denominator[bin_index]);
	if (loop_index == integration_scale*integration_scale*integration_scale - 1) {
		if (all_block_integrations[block_index].integration_numerator[bin_index] == 0) {
			all_block_integrations[block_index].integration_value[bin_index] = 0.0;
		}
		else {
			all_block_integrations[block_index].integration_value[bin_index] = all_block_integrations[block_index].integration_numerator[bin_index] / all_block_integrations[block_index].integration_denominator[bin_index];
			//printf(" final result:%f %f %f\n", all_block_integrations[block_index].integration_numerator[bin_index],
			//	all_block_integrations[block_index].integration_denominator[bin_index], 
			//	all_block_integrations[block_index].integration_value[bin_index]);
		}
	}
}


// »Ö¸´Ò»¸öÌåËØ
__global__
void sgmmRestoreVoxel(unsigned char * raw_result, int width, int depth, int height, int side, int n, int bin_num, sgmmBlock* block_data, double* temp_p, double* temp_p2, sgmmIntegrations* all_block_integrations, int sample_choice) {
	
	//printf("1\n");
	//for every voxel
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (calc_index >= width*depth*height)
		return;
	//global coordinate for every voxel
	int global_height_pos = calc_index / (depth * width);
	int global_depth_pos = (calc_index - global_height_pos * depth * width) / width;
	int global_width_pos = calc_index - global_height_pos * depth * width - global_depth_pos * width;

	//block coordinate
	int height_index = global_height_pos / side;
	int depth_index = global_depth_pos / side;
	int width_index = global_width_pos / side;
	//block index
	int block_index = height_index * ((width / side) * (depth / side)) + depth_index * (width / side) + width_index;
	//if (block_index == 0) {
	//	printf("%d\n", block_data[block_index].bin_num_);
	//}

	double local_pos[3];
	//local coordinate for every voxel
	local_pos[0] = global_width_pos - width_index * side;  //x
	local_pos[1] = global_depth_pos - depth_index * side;  //y
	local_pos[2] = global_height_pos - height_index * side; //z


	double sgmmbi_l[128];
	double P[128];
	for (int i = 0; i <128; i++) {
		sgmmbi_l[i] = 0.0;
		P[i] = 0.0;
	}

	for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
		int real_index = block_data[block_index].bin_indexs_[bin_count];
		//
		if (block_data[block_index].bins_[real_index].probability_ == 0) {
			continue;
		}

		int gauss_count = block_data[block_index].bins_[real_index].gauss_count_;
		sgmmbi_l[real_index] = CalcSGMM(local_pos, gauss_count, block_data, block_index, real_index, n);
	    //if (block_index == 0 && real_index == 100)printf("---%d %f %d\n",real_index,sgmmbi_l[real_index],bin_count);
	}

	//for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
	//	if (sgmmbi_l[bin_count] == 0 || all_block_integrations[block_index].integration_value[bin_count] == 0) {
	//		P[bin_count] = 0.0;
	//		if (block_index == 0 && bin_count == 100)printf("+++%d %f-%f\n",bin_count, sgmmbi_l[bin_count], all_block_integrations[block_index].integration_value[bin_count]);
	//	}
	//	else {
	//		P[bin_count] = sgmmbi_l[bin_count] * (block_data[block_index].bins_[bin_count].probability_ / all_block_integrations[block_index].integration_value[bin_count]); //
	//		if(block_index ==0 && bin_count == 100)printf("&&&%f ", block_data[block_index].bins_[bin_count].probability_);
	//	}
	//	//if (block_index == 0 && bin_count == 100)printf("asdf\n");
	//}

	//////////////////////////////////////////////////////////////////////////
	for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
		int real_index = block_data[block_index].bin_indexs_[bin_count];
		if (sgmmbi_l[real_index] == 0 || all_block_integrations[block_index].integration_value[real_index] == 0) {
			P[real_index] = 0.0;
			//if (block_index == 0 && bin_count == 100)printf("+++%d %f-%f\n", bin_count, sgmmbi_l[real_index], all_block_integrations[block_index].integration_value[real_index]);
		}
		else {
			P[real_index] = sgmmbi_l[real_index] * (block_data[block_index].bins_[real_index].probability_ / all_block_integrations[block_index].integration_value[real_index]); //
			//if (block_index == 0 && bin_count == 100)printf("&&&%f ", block_data[block_index].bins_[bin_count].probability_);
		}
		//if (block_index == 0 && bin_count == 100)printf("asdf\n");
	}
	//////////////////////////////////////////////////////////////////////////
	//
	double sum_p = 0.0;
	for (int i = 0; i < block_data[block_index].bin_num_; i++) {
		int real_index = block_data[block_index].bin_indexs_[i];
		sum_p += P[real_index];
	}
	for (int i = 0; i < block_data[block_index].bin_num_; i++) {
		int real_index = block_data[block_index].bin_indexs_[i];
		P[real_index] /= sum_p;
	}


	//
	int	final_bin_count = 0;
	if (sample_choice == 1) { //
		curandState state;
		curand_init(calc_index, 100, 0, &state);
		double sample = curand_uniform(&state);
		double total_sum = 0.0;
		for (int i = 0; i < block_data[block_index].bin_num_; i++) {
			total_sum += P[i];
			if (total_sum > sample) {
				final_bin_count = i;
				break;
			}
		}
	}
	else if (sample_choice == 2) { //
		double max_p = P[0];
		for (int i = 1; i < block_data[block_index].bin_num_; i++) {
			int ri = block_data[block_index].bin_indexs_[i];
			if (P[ri] > max_p) {
				max_p = P[ri];
				final_bin_count = i;
			}
		}
	}
	else if (sample_choice == 3) { //
		raw_result[calc_index] = 0.0;
		for (int i = 0; i < block_data[block_index].bin_num_; i++) {
			raw_result[calc_index] += P[i] * (i*(256 / bin_num) + 1);
		}
	}
	else if (sample_choice == 4) { //

		final_bin_count = 0;
		double max_p = P[block_data[block_index].bin_indexs_[0]];
		for (int i = 1; i < block_data[block_index].bin_num_; i++) {
			int real_index = block_data[block_index].bin_indexs_[i];
			if (P[real_index] > max_p) {
				max_p = P[real_index];
				final_bin_count = i;
			}
		}

		curandState state;
		curand_init(calc_index, 10, 0, &state);
		for (int i = 0; i < 100; i++) {
			int sample_x = curand_uniform(&state) * block_data[block_index].bin_num_;
			double sample_y = curand_uniform(&state) * max_p;
			int real_index = block_data[block_index].bin_indexs_[sample_x];
			if (sample_y <= P[real_index]) {
				final_bin_count = sample_x;
				break;
			}
		}
	}
	
	if (sample_choice != 3) {
		// Test code begin
		//if (block_index == 3500 && local_pos[0] == 0.0 && local_pos[1] == 0.0 && local_pos[2] == 0.0) {
		//	temp_p[1000] = 3.333333333;
		//	for (int i = 0; i < 128; i++) {
		//		temp_p[i] = P[i];
		//		temp_p2[i] = sgmmbi_l[i];
		//	}
		//}
		// Test code end
		//if (block_data[block_index].bin_indexs_[final_bin_count] == 0) {
		//	printf("%d %d\n", block_index, block_data[block_index].bin_indexs_[final_bin_count]);
		//}
		raw_result[calc_index] = block_data[block_index].bin_indexs_[final_bin_count] * (256 / bin_num) + 1;
		//if (raw_result[calc_index] == 1 && block_index == 0) {
		//	printf("%d %d %d %d\n", block_index, block_data[block_index].bin_indexs_[final_bin_count],final_bin_count,block_data[block_index].bin_num_);
		//}
		//printf("%f\n", raw_result[calc_index]);
	}
}


int restore_raw_by_sgmm(int argc, char ** argv)
{
	//--------constant varible
	const int max_bin_num = 128;
	const int max_sgmm_component_num = 4;
	const int n = 3;
	const int blockSize = 32;
	const int integration_scale = 3;
	//CONSTANT VARIBLE

	std::string data_source;
	std::string disk_address;
	int width;
	int depth;
	int height;
	int side;

	int width_num;
	int depth_num;
	int height_num;
	int block_size;
	int total_size;
	int block_num;

	std::string integration_address;
	std::string src_address;
	std::string sgmm_binary_address;
	std::string result_name;

	unsigned char * raw_src;
	unsigned char * raw_result;
	double * temp_p;
	double * temp_p2;
	int * zero_count;
	int * calc_count;

	std::cout << "----------RESTORE RAW BY SGMM MODULE------------\n";
	std::cout << "input disk address\n";
	std::cin >> disk_address;

	std::cout << "input data name\n";
	std::cin >> data_source;

	integration_address = disk_address + data_source + "_integrations_sgmm";
	src_address = disk_address + data_source + ".raw";
	sgmm_binary_address = disk_address + data_source + "_SGMM_Result.sgmm";
	result_name = disk_address + data_source + "_restored_sgmm.raw";
	std::cout << "input width depth height side (4)\n";
	std::cin >> width >> depth >> height >> side;
	assert(side > 0);


	assert(width%side == 0);
	width_num = width / side;
	assert(depth%side == 0);
	depth_num = depth / side;
	assert(height%side == 0);
	height_num = height / side;

	block_size = side * side *side;
	total_size = width * depth * height;
	block_num = width_num * depth_num * height_num;

	std::cout << data_source << " " << integration_address << " " << sgmm_binary_address << " " << width << " " << depth << " " << height << " " << side << " " << block_num << " " << std::endl;

	std::cout << "Part0: Initializing..." << std::endl;
	unsigned char * raw_result_host = (unsigned char *)malloc(total_size * sizeof(unsigned char));
	unsigned char * temp_p_host = (unsigned char *)malloc(total_size * sizeof(double));
	unsigned char * temp_p2_host = (unsigned char *)malloc(total_size * sizeof(double));

	CUDA_CALL(cudaMalloc(&raw_src, total_size * sizeof(unsigned char)));

	CUDA_CALL(cudaMalloc(&raw_result, total_size * sizeof(unsigned char)));

	CUDA_CALL(cudaMalloc(&temp_p, total_size * sizeof(double)));

	CUDA_CALL(cudaMalloc(&temp_p2, total_size * sizeof(double)));

	CUDA_CALL(cudaMalloc(&all_block_integrations, block_num * sizeof(sgmmIntegrations)));


	for (int i = 0; i < total_size; i++) {
		raw_result_host[i] = 0.0;
		temp_p_host[i] = 0.000;
		temp_p2_host[i] = 0.000;
	}

	CUDA_CALL(cudaMemcpy(raw_result, raw_result_host, total_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(temp_p, temp_p_host, total_size * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(temp_p2, temp_p2_host, total_size * sizeof(double), cudaMemcpyHostToDevice));


	sgmmIntegrations * all_block_integrations_host = (sgmmIntegrations *)malloc(block_num * sizeof(sgmmIntegrations));

	//initializing integrations 
	for (int i = 0; i < block_num; i++) {
		for (int j = 0; j < max_bin_num; j++) {
			all_block_integrations_host[i].integration_value[j] = 0.0;
			all_block_integrations_host[i].integration_denominator[j] = 0.0;
			all_block_integrations_host[i].integration_numerator[j] = 0.0;
		}
	}
	CUDA_CALL(cudaMemcpy(all_block_integrations, all_block_integrations_host, block_num * sizeof(sgmmIntegrations), cudaMemcpyHostToDevice));

	sgmmBlock* block_data;
	//cudaMallocManaged(&block_data, block_num * sizeof(Block));
	block_data = (sgmmBlock*)malloc(sizeof(sgmmBlock)*block_num);
	if (block_data == nullptr) {
		std::cout << "can not allocate memory for block. ";
		std::cout << sizeof(sgmmBlock)*block_num / 1024 / 1024 << "m are needed.\n";
		return 0;
	}
	std::cout << "Part1: Reading SGMMs..." << std::endl;
	std::ifstream f_sgmm(sgmm_binary_address, std::ios::binary);
	if (f_sgmm.is_open() == false) {
		std::cout << "can not open file\n";
		return 0;
	}
	for (int block_index = 0; block_index < block_num; block_index++) {

		f_sgmm.read((char*)&(block_data[block_index].bin_num_), sizeof(unsigned char));

		assert(block_data[block_index].bin_num_ >= 0);
		assert(block_data[block_index].bin_num_ <= max_bin_num);
		for (int bin_count = 0; bin_count < block_data[block_index].bin_num_; bin_count++) {
			f_sgmm.read((char*)&(block_data[block_index].bin_indexs_[bin_count]), sizeof(unsigned char));
			//std::cout << "blockcount:"<<block_data[block_index].bin_indexs_[bin_count] << " ";
			int real_bin_index = block_data[block_index].bin_indexs_[bin_count];
			assert(real_bin_index >= 0);
			assert(real_bin_index < max_bin_num);
			//std::cout << "real_inex"<<real_bin_index;
			f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].probability_), sizeof(float));
			f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gauss_count_), sizeof(unsigned char));
			//std::cout << "prob:"<<block_data[block_index].bins_[real_bin_index].probability_ << " gauss count:" << block_data[block_index].bins_[real_bin_index].gauss_count_<< std::endl;
			assert(block_data[block_index].bins_[real_bin_index].gauss_count_ >= 0);
			assert(block_data[block_index].bins_[real_bin_index].gauss_count_ <= 4);
			for (int gauss_index = 0; gauss_index < block_data[block_index].bins_[real_bin_index].gauss_count_; gauss_index++) {
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].weight_), sizeof(float));
				for (int i = 0; i < 3; i++) {
					f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].mean_[i]), sizeof(float));
				}
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[0]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[1]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[2]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[4]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[5]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[8]), sizeof(float));
				block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[3] = block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[1];
				block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[6] = block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[2];
				block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[7] = block_data[block_index].bins_[real_bin_index].gausses_[gauss_index].covariance_[5];
			}
		}
	}
	f_sgmm.close();
	sgmmBlock * block_data_device;
	CUDA_CALL(cudaMalloc(&block_data_device, block_num*(sizeof(sgmmBlock))));
	CUDA_CALL(cudaMemcpy(block_data_device, block_data, block_num*(sizeof(sgmmBlock)), cudaMemcpyHostToDevice));
	free(block_data);
	block_data = block_data_device;

	// Part2~4: 
	int numBlocks;
	clock_t start, finish;
	int read_from;
	std::cout << std::endl << "Choose a way to get integrations, input 1 to calc or 2 to read." << std::endl;
	std::cin >> read_from;
	if (read_from == 1) {
		// Part2~4:
		// Part2:
		std::cout << std::endl << "Part2: Calculating integrations numerator..." << std::endl;

		/*NOTICE:  
		 *blockSize and the numBlocks should not be larger than the limitation of
		 * capability of the display card.Please run related CUDA API
		 *to get the proper size
		*/
		numBlocks = (block_num * max_bin_num + blockSize - 1) / blockSize;
		std::cout << "thread block num:" << numBlocks << " thread block size:" << blockSize << std::endl;
		start = clock();
		CalcIntegrationsNumerator <<< numBlocks, blockSize >> > (all_block_integrations, block_data, side, n, block_num, max_bin_num, temp_p, temp_p2);
		CUDA_CALL(cudaDeviceSynchronize());
		finish = clock();

		//////////////////////////////////////////////////////////////////////////
		///output intermediate result for testing
		//
		std::ofstream inter_result("e:/testinfo/numeratorinfo.sgmm", std::ios::binary);
		if (inter_result.is_open() == true) {
			CUDA_CALL(cudaMemcpy(all_block_integrations_host, all_block_integrations, block_num*(sizeof(sgmmIntegrations)), cudaMemcpyDeviceToHost));
			inter_result.write((const char *)all_block_integrations_host, block_num*(sizeof(sgmmIntegrations)));
		}
		//////////////////////////////////////////////////////////////////////////


		std::cout << "Calculating numerator time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "s" << std::endl;

		int start_index = 0;
		std::cout << std::endl << "Part3: Calculating integrations denominator..." << std::endl;
		for (int loop_index = start_index; loop_index < integration_scale * integration_scale * integration_scale; loop_index++) {
			std::cout << "Calculating block " << loop_index << "..." << std::endl;

			start = clock();
			CalcIntegrationsDenominator << <numBlocks, blockSize >> > (all_block_integrations, block_data, side, n, block_num, max_bin_num, loop_index, integration_scale, temp_p, temp_p2);
			CUDA_CALL(cudaDeviceSynchronize());
			finish = clock();

			CUDA_CALL(cudaMemcpy(all_block_integrations_host, all_block_integrations, block_num*(sizeof(sgmmIntegrations)), cudaMemcpyDeviceToHost));

			std::cout << "Calculating denominator time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
			//////////////////////////////////////////////////////////////////////////
			//std::ofstream f_temp_integration_out("e:/testinfo/i_" + Int2String(loop_index), std::ios::binary);
			//f_temp_integration_out.write((char *)all_block_integrations_host, block_num * sizeof(sgmmIntegrations));
			//f_temp_integration_out.close();
			//////////////////////////////////////////////////////////////////////////
			//Mark(Int2String(loop_index));
		}
		// Part4:

		std::cout << std::endl << "Part4: Saving integrations..." << std::endl;
		std::ofstream f_out(integration_address, std::ios::binary);
		for (int i = 0; i < block_num; i++) {
			for (int j = 0; j < max_bin_num; j++) {
				f_out.write((char*)&(all_block_integrations_host[i].integration_value[j]), sizeof(float));
			}
		}
	}
	else if (read_from == 2) {
		// Part2~4:
		std::cout << std::endl << "Part2~4: Reading integrations..." << std::endl;
		std::ifstream f_in(integration_address, std::ios::binary);
		if (f_in.is_open() == false) {
			std::cout << "can not open integrations file\n";
			return 0;
		}
		for (int i = 0; i < block_num; i++) {
			for (int j = 0; j < max_bin_num; j++) {
				f_in.read((char*)&(all_block_integrations_host[i].integration_value[j]), sizeof(float));
			}
		}
		f_in.close();
		CUDA_CALL(cudaMemcpy(all_block_integrations, all_block_integrations_host, block_num * sizeof(sgmmIntegrations), cudaMemcpyHostToDevice));
	}
	else {
		exit(1);
	}

	// Part5: ÖØ¹¹rawÊý¾Ý

	std::cout << std::endl << "Part5: Restoring data..." << std::endl;
	std::cout << std::endl << "Choose a way to restore data, input 1 to sample randomly, 2 to sample by max, 3 to sample by integration, 4 to sample by rejection method." << std::endl;
	int sample_choice;
	std::cin >> sample_choice;
	if (sample_choice != 1 && sample_choice != 2 && sample_choice != 3 && sample_choice != 4) {
		std::cout << "Illegal input." << std::endl;
		exit(1);
	}
	numBlocks = (((width / side)*side) * ((depth / side)*side) * ((height / side)*side) + blockSize - 1) / blockSize; //
	std::cout << "thread block num:" << numBlocks << " thread block size:" << blockSize << std::endl;

	start = clock();
	sgmmRestoreVoxel<<<numBlocks, blockSize >>>(raw_result, width, depth, height,
		side, n, max_bin_num, block_data,
		temp_p, temp_p2, all_block_integrations, sample_choice);
	CUDA_CALL(cudaDeviceSynchronize());
	finish = clock();

	std::cout << "Restoring time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
	//Test code Begin
	//while (true) {
	//	int index;
	//	std::cin >> index;
	//	if (index == -1) break;
	//	PrintData(index, raw_result);
	//}
	//Test code End

	//// Part6: »¹Ô­ÎªrawÎÄ¼þ
	std::cout << std::endl << "Part6: Saving raw file..." << std::endl;
	std::ofstream f_result(result_name, std::ios::binary);
	CUDA_CALL(cudaMemcpy(raw_result_host, raw_result, total_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	if (f_result.is_open() == false) {
		std::cout << "can not open result file\n";
		exit(1);
	}
	f_result.write((const char*)raw_result_host, total_size);
	f_result.close();

	//Creating .vifo file
	create_vifo_file(disk_address, disk_address+"_restored_sgmm", width, depth, height);

	CUDA_CALL(cudaFree(raw_result));
	CUDA_CALL(cudaFree(temp_p));
	CUDA_CALL(cudaFree(temp_p2));
	CUDA_CALL(cudaFree(block_data));

	//
	std::cin.get();
	return 0;
}
