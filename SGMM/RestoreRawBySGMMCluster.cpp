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

#define DEBUG_RESTOREVOXEL_KERNEL
//#define DEBUG_NUMERATOR_KERNEL
//#define DEBUG_DENOMINATOR_KERNEL




//#define MONTE_CARLO
// 参数控制区域 begin


struct Point3d {
	pos_type x;
	pos_type y;
	pos_type z;
	Point3d(pos_type xx = 0, pos_type yy = 0, pos_type zz = 0) :
		x{ xx }, y{ yy }, z{ zz } {}
	Point3d operator+(const Point3d & p)const {
		return Point3d{ x + p.x, y + p.y, z + p.z };
	}
	friend std::ostream & operator<<(std::ostream & os, const Point3d & p) {
		os << p.x << " " << p.y << " " << p.z;
		return os;
	}
	bool operator<(const Point3d & p)const {
		return (x < p.x&&y < p.y&&z < p.z);
	}
	bool operator<=(const Point3d & p)const {
		return (x <= p.x&&y <= p.y&&z <= p.z);
	}
	bool operator>=(const Point3d & p)const {
		return !(*this < p);
	}
	bool operator>(const Point3d & p)const {
		return !(*this <= p);
	}
	bool operator==(const Point3d & p)const {
		return (x == p.x&&y == p.y&&z == p.z);
	}
	bool operator!=(const Point3d & p)const {
		return !(*this == p);
	}
};


struct debug_block_info {
	Point3d min_point;
	Point3d max_point;
	int cluster_num;
	int gauss_count[MAX_CLUSTER_NUM];
	debug_block_info() :min_point{}, max_point{}, cluster_num{}, gauss_count{} {}
};



// 整数转字符串
static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}


// 保存第index个块到d盘（测试函数）
//void PrintData(int index, unsigned char* data) {
//	std::ofstream f_temp;
//	f_temp.open("d:/restored_block_" + Int2String(index) + ".txt");
//
//	int height_index = index / (width_num * depth_num);
//	int depth_index = (index - height_index * width_num * depth_num) / width_num;
//	int width_index = index - height_index * width_num * depth_num - depth_index * width_num;
//
//	for (int z = 0; z < side; z++) {
//		for (int y = 0; y < side; y++) {
//			for (int x = 0; x < side; x++) {
//				int final_index_z = height_index * side + z;
//				int final_index_y = depth_index * side + y;
//				int final_index_x = width_index * side + x;
//				int final_index = final_index_z * width * depth + final_index_y * width + final_index_x;
//				f_temp << (int)(data[final_index]) << std::endl;
//			}
//		}
//	}
//	f_temp.close();
//}

//debug for integrations




// 标记函数（测试函数）
static void Mark(std::string event_name) {
	// 做一个标记
	std::ofstream f_temp;
	f_temp.open("d:/" + event_name + ".txt");
	f_temp.close();
}


// 计算A^T * B * C
__device__
double MulMatrix(double * A, float * B, double * C) {
	return (A[0] * B[0] + A[1] * B[3] + A[2] * B[6]) * C[0] + (A[0] * B[1] + A[1] * B[4] + A[2] * B[7]) * C[1] + (A[0] * B[2] + A[1] * B[5] + A[2] * B[8]) * C[2];
}


// 计算一个gmm分支
__device__
double CalcGMM(float* mean, float* inv_cov, double determinant, double* local_pos, int n) {
	double diff[3]; //当前坐标和平均值的差值
					//printf("%lf\n", determinant);
	for (int i = 0; i < 3; i++) {
		diff[i] = local_pos[i] - mean[i];
	}
	//printf("%f", MulMatrix(diff, inv_cov, diff));
	return 1.0 / sqrt(8 * M_PI * M_PI * M_PI *determinant) * __expf(-0.5 * MulMatrix(diff, inv_cov, diff));
	//return 1.0 / sqrt(pow(2 * M_PI, n)*determinant) *1.0;
	//return 1.0 * __expf(-0.5 * MulMatrix(diff, inv_cov, diff));
	//return 0;
}


// 计算local_pos位置处的sgmm（概率密度）
__device__
double CalcSGMM(double* local_pos, int gauss_count, sgmmClusterBlock* block_data_device, int block_index, int cluster_index, int n) {
	double sgmm_result = 0.0;	
	for (int gauss_index = 0; gauss_index < gauss_count; gauss_index++) {
		double added_value = block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].weight_
			* CalcGMM(block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].mean_,
				block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_,
				block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].determinant_,
				local_pos, n);
		sgmm_result += added_value; //local_pos坐标下的sgmm值
		//printf("%d %f %f\n",cluster_index, block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].weight_,added_value);
		//printf("sgmmcluster:%f %f %f %f\n", local_pos[0], local_pos[1], local_pos[2], added_value);
	}
	//printf("sgmmcluster:%f %f %f %f\n",local_pos[0],local_pos[1],local_pos[2], sgmm_result);
	return sgmm_result;
}


// 计算block_index块内cluster_index这个cluster的sgmm在块内的积分的分子部分
//__global__
//void CalcIntegrationsNumerator(Integrations* all_block_integrations, Block* block_data, int side, int n, int block_num, int max_cluster_num, double * temp_p, double * temp_p2) {
//	double offset = 0.000;
//	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
//	if (calc_index >= block_num * max_cluster_num) return;
//
//
//	//(1)
//	int block_index = calc_index / max_cluster_num;
//	int cluster_index = calc_index - block_index * max_cluster_num;
//
//	if (cluster_index >= block_data[block_index].cluster_num_) {
//		return;
//	}
//
//
//	if (block_data[block_index].clusters_[cluster_index].gauss_count_ == 0) {
//		all_block_integrations[block_index].integration_numerator[cluster_index] = 0.0;
//		return;
//	}
//
//	double numerator = 0.0;
//	// 计算SGMM在小方块内的积分
//	for (int z = 0; z < side; z++) {
//		for (int y = 0; y < side; y++) {
//			for (int x = 0; x < side; x++) {
//				double local_pos[3] = { x + offset,y + offset,z + offset };
//				int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
//				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
//				numerator += sgmm;
//			}
//		}
//	}
//
//	all_block_integrations[block_index].integration_numerator[cluster_index] = numerator;
//}
__global__
void CalcIntegrationsNumerator(sgmmClusterIntegrations* all_block_integrations, sgmmClusterBlock* block_data, int n, int block_num, int max_cluster_num, double * temp_p, double * temp_p2, Point3d * debug_points1_device, Point3d * debug_points2_device, Point3d * min_points, Point3d * max_points) {
	double offset = 0.000;
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
	if (calc_index >= block_num * max_cluster_num) return;
	//printf("%d ", calc_index);

	//(1)
	int block_index = calc_index / max_cluster_num;
	int cluster_index = calc_index - block_index * max_cluster_num;
	//printf("%d ", block_index);

	debug_points1_device[block_index] = min_points[block_index];
	debug_points2_device[block_index] = max_points[block_index];

	if (cluster_index >= block_data[block_index].cluster_num_) {
		//printf("a %d %d %d \n", block_index, block_data[block_index].cluster_num_, cluster_index);
		return;
	}


	if (block_data[block_index].clusters_[cluster_index].gauss_count_ == 0) {
		all_block_integrations[block_index].integration_numerator[cluster_index] = 0.0;
		//printf("%d\n", block_index);
		return;
	}

	double numerator = 0.0;
	//debug
#ifdef DEBUG_NUMERATOR_KERNEL

	//temp_p[block_index] = block_index;
	//printf("%d %d %d %d %d %d %d\n", min_points[block_index].x, min_points[block_index].y, min_points[block_index].z, max_points[block_index].x, max_points[block_index].y, max_points[block_index].z,block_index);
#endif


	// 计算SGMM在小方块内的积分
	int zside = max_points[block_index].z - min_points[block_index].z;
	int yside = max_points[block_index].y - min_points[block_index].y;
	int xside = max_points[block_index].x - min_points[block_index].x;

	for (auto z = 0; z < zside; z++) {
		for (auto y = 0; y < yside; y++) {
			for (auto x = 0; x < xside; x++) {
				double local_pos[3] = { x + offset, y + offset, z + offset };
				int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
				numerator += sgmm;
			}
		}
	}

	all_block_integrations[block_index].integration_numerator[cluster_index] = numerator;

}

__global__
void MonteCarloIntegrations(sgmmClusterIntegrations* all_block_integrations,
	sgmmClusterBlock* block_data, 
	int n, 
	int block_num,
	int max_cluster_num, 
	double * temp_p,
	double * temp_p2, 
	Point3d * debug_points1_device,
	Point3d * debug_points2_device, 
	Point3d * min_points, 
	Point3d * max_points) {
	double offset = 0.000;
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
	if (calc_index >= block_num * max_cluster_num) return;
	//printf("%d ", calc_index);

	//(1)
	int block_index = calc_index / max_cluster_num;
	int cluster_index = calc_index - block_index * max_cluster_num;
	//printf("%d ", block_index);

	debug_points1_device[block_index] = min_points[block_index];
	debug_points2_device[block_index] = max_points[block_index];

	if (cluster_index >= block_data[block_index].cluster_num_) {
		//printf("a %d %d %d \n", block_index, block_data[block_index].cluster_num_, cluster_index);
		return;
	}


	if (block_data[block_index].clusters_[cluster_index].gauss_count_ == 0) {
		all_block_integrations[block_index].integration_numerator[cluster_index] = 0.0;
		//printf("%d\n", block_index);
		return;
	}

	double numerator = 0.0;
	//debug
#ifdef DEBUG_NUMERATOR_KERNEL

	//temp_p[block_index] = block_index;
	//printf("%d %d %d %d %d %d %d\n", min_points[block_index].x, min_points[block_index].y, min_points[block_index].z, max_points[block_index].x, max_points[block_index].y, max_points[block_index].z,block_index);
#endif


	// 计算SGMM在小方块内的积分
	int zside = max_points[block_index].z - min_points[block_index].z;
	int yside = max_points[block_index].y - min_points[block_index].y;
	int xside = max_points[block_index].x - min_points[block_index].x;

	for (auto z = 0; z < zside; z++) {
		for (auto y = 0; y < yside; y++) {
			for (auto x = 0; x < xside; x++) {
				double local_pos[3] = { x + offset, y + offset, z + offset };
				int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
				numerator += sgmm;
				//printf("%d\n", (int)numerator);
			}
		}
	}

	all_block_integrations[block_index].integration_value[cluster_index] = numerator;

}


// 计算block_index块内cluster_index这个cluster的sgmm在块内的积分的分母部分
//__global__
//void CalcIntegrationsDenominator(Integrations* all_block_integrations, Block* block_data,int side, int n, int block_num, int max_cluster_num, int loop_index, int integration_scale, double * temp_p, double * temp_p2) {
//	
//	double offset = 0.000;
//	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
//	if (calc_index >= block_num * max_cluster_num) return;
//
//
//	(1)
//	int block_index = calc_index / max_cluster_num;
//	int cluster_index = calc_index - block_index * max_cluster_num;
//
//	if (cluster_index >= block_data[block_index].cluster_num_) {
//		return;
//	}
//
//	int z_index = loop_index / (integration_scale*integration_scale);
//	int y_index = (loop_index - z_index * (integration_scale*integration_scale)) / integration_scale;
//	int x_index = loop_index - z_index * (integration_scale*integration_scale) - y_index * integration_scale;
//
//	if (block_data[block_index].clusters_[cluster_index].gauss_count_ == 0) {
//		all_block_integrations[block_index].integration_denominator[cluster_index] += 0.0;
//		if (loop_index == integration_scale * integration_scale * integration_scale - 1) {
//			if (all_block_integrations[block_index].integration_denominator[cluster_index] == 0) {
//				all_block_integrations[block_index].integration_value[cluster_index] = 0.0;
//			}
//			else {
//				all_block_integrations[block_index].integration_value[cluster_index] = all_block_integrations[block_index].integration_numerator[cluster_index] / all_block_integrations[block_index].integration_denominator[cluster_index];
//			}
//		}
//		return;
//	}
//
//	double denominator = 0.0;
//	for (int z = (-1 + z_index) * side; z < (0 + z_index) * side; z++) {
//		for (int y = (-1 + y_index) * side; y < (0 + y_index) * side; y++) {
//			for (int x = (-1 + x_index) * side; x < (0 + x_index) * side; x++) {
//				double local_pos[3] = { x + offset,y + offset,z + offset };
//				int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
//				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
//				denominator += sgmm;
//			}
//		}
//	}
//
//	if (denominator == 0) {
//		all_block_integrations[block_index].integration_denominator[cluster_index] += 0.0;
//		if (loop_index == integration_scale*integration_scale*integration_scale - 1) {
//			if (all_block_integrations[block_index].integration_denominator[cluster_index] == 0) {
//				all_block_integrations[block_index].integration_value[cluster_index] = 0.0;
//			}
//			else {
//				all_block_integrations[block_index].integration_value[cluster_index] = all_block_integrations[block_index].integration_numerator[cluster_index] / all_block_integrations[block_index].integration_denominator[cluster_index];
//			}
//		}
//		return;
//	}
//	else {
//		all_block_integrations[block_index].integration_denominator[cluster_index] += denominator;
//		if (loop_index == integration_scale*integration_scale*integration_scale - 1) {
//			if (all_block_integrations[block_index].integration_denominator[cluster_index] == 0) {
//				all_block_integrations[block_index].integration_value[cluster_index] = 0.0;
//			}
//			else {
//				all_block_integrations[block_index].integration_value[cluster_index] = all_block_integrations[block_index].integration_numerator[cluster_index] / all_block_integrations[block_index].integration_denominator[cluster_index];
//			}
//		}
//		return;
//	}
//}
__global__
void CalcIntegrationsDenominator(sgmmClusterIntegrations* all_block_integrations, sgmmClusterBlock* block_data, int n, int block_num, int max_cluster_num, int loop_index, int integration_scale, double * temp_p, double * temp_p2, Point3d* debug_points1_device, Point3d * debug_points2_device, Point3d * min_points, Point3d * max_points) {

	double offset = 0.000;
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
	if (calc_index >= block_num * max_cluster_num) return;


	//
	int block_index = calc_index / max_cluster_num;
	int cluster_index = calc_index - block_index * max_cluster_num;

	if (cluster_index >= block_data[block_index].cluster_num_) {
		return;
	}

	int z_index = loop_index / (integration_scale*integration_scale);
	int y_index = (loop_index - z_index * (integration_scale*integration_scale)) / integration_scale;
	int x_index = loop_index - z_index * (integration_scale*integration_scale) - y_index * integration_scale;

	if (block_data[block_index].clusters_[cluster_index].gauss_count_ == 0) {
		all_block_integrations[block_index].integration_denominator[cluster_index] += 0.0;
		if (loop_index == integration_scale * integration_scale * integration_scale - 1) {
			if (all_block_integrations[block_index].integration_denominator[cluster_index] == 0) {
				all_block_integrations[block_index].integration_value[cluster_index] = 0.0;
			}
			else {
				all_block_integrations[block_index].integration_value[cluster_index] = all_block_integrations[block_index].integration_numerator[cluster_index] / all_block_integrations[block_index].integration_denominator[cluster_index];
			}
		}
		return;
	}
	//debug
#ifdef DEBUG_DENOMINATOR_KERNEL
	debug_points1_device[block_index] = min_points[block_index];
	debug_points2_device[block_index] = max_points[block_index];
#endif


	double denominator = 0.0;
	int zside = max_points[block_index].z - min_points[block_index].z;
	int yside = max_points[block_index].y - min_points[block_index].y;
	int xside = max_points[block_index].x - min_points[block_index].x;
	for (int z = (-1 + z_index) * zside; z < (0 + z_index) * zside; z++) {
		for (int y = (-1 + y_index) * yside; y < (0 + y_index) * yside; y++) {
			for (int x = (-1 + x_index) * xside; x < (0 + x_index) * xside; x++) {
				double local_pos[3] = { x + offset, y + offset, z + offset };
				int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
				double sgmm = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
				denominator += sgmm;
			}
		}
	}

	if (denominator == 0) {
		all_block_integrations[block_index].integration_denominator[cluster_index] += 0.0;
		if (loop_index == integration_scale*integration_scale*integration_scale - 1) {
			if (all_block_integrations[block_index].integration_denominator[cluster_index] == 0) {
				all_block_integrations[block_index].integration_value[cluster_index] = 0.0;
			}
			else {
				all_block_integrations[block_index].integration_value[cluster_index] = all_block_integrations[block_index].integration_numerator[cluster_index] / all_block_integrations[block_index].integration_denominator[cluster_index];
			}
		}
		return;
	}
	else {
		all_block_integrations[block_index].integration_denominator[cluster_index] += denominator;
		if (loop_index == integration_scale*integration_scale*integration_scale - 1) {
			if (all_block_integrations[block_index].integration_denominator[cluster_index] == 0) {
				all_block_integrations[block_index].integration_value[cluster_index] = 0.0;
			}
			else {
				all_block_integrations[block_index].integration_value[cluster_index] = all_block_integrations[block_index].integration_numerator[cluster_index] / all_block_integrations[block_index].integration_denominator[cluster_index];
			}
		}
		return;
	}
}

// 恢复一个体素
//__global__
//void RestoreVoxel(unsigned char * raw_result, int width, int depth, int height, int side, int n, int max_cluster_num, int width_num, int depth_num, int height_num,
//	Block* block_data, double* temp_p, double* temp_p2, Integrations* all_block_integrations, int sample_choice) {
//	//计算体素的索引
//	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
//
//	//计算全局坐标
//	int global_height_pos = calc_index / (depth * width);
//	int global_depth_pos = (calc_index - global_height_pos * depth * width) / width;
//	int global_width_pos = calc_index - global_height_pos * depth * width - global_depth_pos * width;
//	
//	//计算块的索引
//	int height_index = global_height_pos / side;
//	int depth_index = global_depth_pos / side;
//	int width_index = global_width_pos / side;
//	//(1)
//	int block_index = height_index * (width_num * depth_num) + depth_index * width_num + width_index;
//
//	if (block_data[block_index].cluster_num_ == 0) {
//		raw_result[calc_index] = 0.0;
//		return;
//	}
//
//	//计算块内坐标   (2)
//	double local_pos[3];
//	local_pos[0] = global_width_pos - width_index * side;  //x
//	local_pos[1] = global_depth_pos - depth_index * side;  //y
//	local_pos[2] = global_height_pos - height_index * side; //z
//
//	//计算所有cluster的概率
//	double sgmmbi_l[MAX_CLUSTER_NUM];
//	double P[MAX_CLUSTER_NUM];
//	for (int i= 0; i < max_cluster_num; i++) {
//		sgmmbi_l[i] = 0.0;
//		P[i] = 0.0;
//	}
//
//	for (int cluster_index=0; cluster_index < block_data[block_index].cluster_num_; cluster_index++) {
//		//如果cluster的probability为0，则sgmmbi_i[i]为0，直接跳过
//		if (block_data[block_index].clusters_[cluster_index].probability_ == 0) {
//			continue;
//		}
//
//		//计算SGMM（概率密度）
//		int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
//		sgmmbi_l[cluster_index] = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
//	}
//
//
//	for (int cluster_index = 0; cluster_index < block_data[block_index].cluster_num_; cluster_index++) {
//		if (sgmmbi_l[cluster_index] == 0 || all_block_integrations[block_index].integration_value[cluster_index] == 0) {
//			P[cluster_index] = 0.0;
//		}
//		else {
//			P[cluster_index] = sgmmbi_l[cluster_index] * (block_data[block_index].clusters_[cluster_index].probability_ / all_block_integrations[block_index].integration_value[cluster_index]); // 除以SGMM在块内的积分
//		}
//	}
//
//	//归一化P
//	double sum_p = 0.0;
//	for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
//		sum_p += P[i];
//	}
//	for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
//		P[i] /= sum_p;
//	}
//
//
//	//采样
//	int	final_cluster_index = 0;
//	if (sample_choice == 1) { //方法一：随机采样
//		curandState state;
//		curand_init(calc_index, 100, 0, &state);
//		double sample = curand_uniform(&state);
//		double total_sum = 0.0;
//		for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
//			total_sum += P[i];
//			if (total_sum > sample) {
//				final_cluster_index = i;
//				break;
//			}
//		}
//	}
//	else if(sample_choice == 2){ //方法二：取概率最大采样
//		double max_p = P[0];
//		for (int i = 1; i < block_data[block_index].cluster_num_; i++) {
//			if (P[i] > max_p) {
//				max_p = P[i];
//				final_cluster_index = i;
//			}
//		}
//	}
//	else if (sample_choice == 3) { // 方法三：取积分采样
//		raw_result[calc_index] = 0.0;
//		for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
//			raw_result[calc_index] += P[i] * (block_data[block_index].clusters_[i].sample_value_);
//		}
//	}
//	else if (sample_choice == 4) { // 方法四：蒙特卡洛拒绝采样
//		double max_p = P[0];
//		final_cluster_index = 0;
//		for (int i = 1; i < block_data[block_index].cluster_num_; i++) {
//			if (P[i] > max_p) {
//				max_p = P[i];
//				final_cluster_index = i;
//			}
//		}
//
//		curandState state;
//		curand_init(calc_index, 10, 0, &state);
//		for(int i=0; i<100; i++){
//			int sample_x = curand_uniform(&state) * block_data[block_index].cluster_num_;
//			double sample_y = curand_uniform(&state) * max_p;
//			if (sample_y <= P[sample_x]) {
//				final_cluster_index = sample_x;
//				break;
//			}
//		}
//	}
//
//	if (sample_choice != 3) {
//		raw_result[calc_index] = block_data[block_index].clusters_[final_cluster_index].sample_value_;
//	}
//}
__global__
static void RestoreVoxel(unsigned char * raw_result,
	int width,
	int depth,
	int height,
	int n,
	int max_cluster_num,
	sgmmClusterBlock* block_data,
	double* temp_p,
	double* temp_p2,
	sgmmClusterIntegrations* all_block_integrations,
	int sample_choice,
	id_type * id_table,
	Point3d * min_points,
	Point3d * max_points,
	debug_block_info * debug_block,
	int total_size) {
	//计算体素的索引
	long calc_index = blockIdx.x * blockDim.x + threadIdx.x; //计算单元的索引
	if (calc_index >= total_size)return;
	//printf("%d\n", calc_index);

	//计算全局坐标
	int global_height_pos = calc_index / (depth * width);
	int global_depth_pos = (calc_index - global_height_pos * depth * width) / width;
	int global_width_pos = calc_index - global_height_pos * depth * width - global_depth_pos * width;

	//计算块的索引

	//(1)
	int block_index = 0;
	block_index = id_table[calc_index];
#ifdef DEBUG_RESTOREVOXEL_KERNEL
	{
		temp_p[calc_index] = block_index;
		debug_block[block_index].min_point = min_points[block_index];
		debug_block[block_index].max_point = max_points[block_index];
		int cluster_num = block_data[block_index].cluster_num_;
		debug_block[block_index].cluster_num = cluster_num;
		for (int i = 0; i < cluster_num; i++) {
			debug_block[block_index].gauss_count[i] = block_data[block_index].clusters_[i].gauss_count_;
		}
	}
#endif 
	//printf("%d %d\n", calc_index, block_index);
	//printf("%d\n", block_index);
	//block_index = 0;
	if (block_data[block_index].cluster_num_ == 0) {
		raw_result[calc_index] = 0.0;
		return;
	}

	//计算块内坐标   (2)
	double local_pos[3];
	local_pos[0] = global_width_pos - min_points[block_index].x; //x
	local_pos[1] = global_depth_pos - min_points[block_index].y;  //y
	local_pos[2] = global_height_pos - min_points[block_index].z; //z

																  //计算所有cluster的概率
	double sgmmbi_l[MAX_CLUSTER_NUM];
	double P[MAX_CLUSTER_NUM];
	for (int i = 0; i < max_cluster_num; i++) {
		sgmmbi_l[i] = 0.0;
		P[i] = 0.0;
	}

	for (int cluster_index = 0; cluster_index < block_data[block_index].cluster_num_; cluster_index++) {
		//如果cluster的probability为0，则sgmmbi_i[i]为0，直接跳过
		if (block_data[block_index].clusters_[cluster_index].probability_ == 0) {
			continue;
		}

		//计算SGMM（概率密度）
		int gauss_count = block_data[block_index].clusters_[cluster_index].gauss_count_;
		sgmmbi_l[cluster_index] = CalcSGMM(local_pos, gauss_count, block_data, block_index, cluster_index, n);
	}


	for (int cluster_index = 0; cluster_index < block_data[block_index].cluster_num_; cluster_index++) {
		if (sgmmbi_l[cluster_index] == 0 || all_block_integrations[block_index].integration_value[cluster_index] == 0) {
			P[cluster_index] = 0.0;
		}
		else {
			P[cluster_index] = sgmmbi_l[cluster_index] * (block_data[block_index].clusters_[cluster_index].probability_ / all_block_integrations[block_index].integration_value[cluster_index]); // 除以SGMM在块内的积分
		}
	}

	//归一化P
	double sum_p = 0.0;
	for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
		sum_p += P[i];
	}
	for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
		P[i] /= sum_p;
	}


	//采样
	int	final_cluster_index = 0;
	if (sample_choice == 1) { //方法一：随机采样
		curandState state;
		curand_init(calc_index, 100, 0, &state);
		double sample = curand_uniform(&state);
		double total_sum = 0.0;
		for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
			total_sum += P[i];
			if (total_sum > sample) {
				final_cluster_index = i;
				break;
			}
		}
	}
	else if (sample_choice == 2) { //方法二：取概率最大采样
		double max_p = P[0];
		for (int i = 1; i < block_data[block_index].cluster_num_; i++) {
			if (P[i] > max_p) {
				max_p = P[i];
				final_cluster_index = i;
			}
		}
	}
	else if (sample_choice == 3) { // 方法三：取积分采样
		raw_result[calc_index] = 0.0;
		for (int i = 0; i < block_data[block_index].cluster_num_; i++) {
			raw_result[calc_index] += P[i] * (block_data[block_index].clusters_[i].sample_value_);
		}
	}
	else if (sample_choice == 4) { // 方法四：蒙特卡洛拒绝采样
		double max_p = P[0];
		final_cluster_index = 0;
		for (int i = 1; i < block_data[block_index].cluster_num_; i++) {
			if (P[i] > max_p) {
				max_p = P[i];
				final_cluster_index = i;
			}
		}

		curandState state;
		curand_init(calc_index, 10, 0, &state);
		for (int i = 0; i < 100; i++) {
			int sample_x = curand_uniform(&state) * block_data[block_index].cluster_num_;
			double sample_y = curand_uniform(&state) * max_p;
			if (sample_y <= P[sample_x]) {
				final_cluster_index = sample_x;
				break;
			}
		}
	}

	if (sample_choice != 3) {
		raw_result[calc_index] = block_data[block_index].clusters_[final_cluster_index].sample_value_;
	}
}
__host__
int restoreRawBySGMMCluster(int argc, char ** argv)
{
	int width;
	int depth;
	int height;
	std::string data_source;
	std::string disk_address;
	const int blockSize = 1024; // CUDA块内kernel数

	std::string integration_cluster_address;
	std::string sgmm_binary_address;
	std::string result_name;

	const int n = 3; // 数据维度
	const int integration_scale = 3;

	int block_num;
	int total_size;
	unsigned char * raw_src;
	unsigned char * raw_result;
	double * temp_p = nullptr; //
	double * temp_p2 = nullptr; // 
	int * zero_count;
	int * calc_count;

	sgmmClusterIntegrations* all_block_integrations;

#if defined(DEBUG_RESTOREVOXEL_KERNEL)||defined(DEBUG_DENOMINATOR_KERNEL)||defined(DEBUG_NUMERATOR_KERNEL)

	Point3d * debug_points1_device;
	Point3d * debug_points2_device;
#endif

	//
	std::cout << "-----------RESOTRE RAW BY SGMM CLUSTER MODULE-----------\n";
	// Part0: 初始化
	std::cout << "input data address\n";
	std::cin >> disk_address;

	std::cout << "input data name\n";
	std::cin >> data_source;

	integration_cluster_address = disk_address + data_source + "_integrations_sgmm_cluster";
	sgmm_binary_address = disk_address + data_source + "_SGMM_Cluster_Result.sgmm";
	result_name = disk_address + data_source + "_restored_sgmm_cluster" + ".raw";

	std::cout << "input width depth height (3)\n";
	std::cin >> width >> depth >> height;

	//读取id2boundingbox文件
	std::ifstream bounding_box_file(disk_address + data_source + std::string{ ".reoc" });
	if (bounding_box_file.is_open() == false) {
		std::cout << "can not open id2boungdingbox failed\n";
		return 0;
	}
	bounding_box_file >> block_num;
	//这里指的是数据的min_points max_points

	Point3d * min_points = new Point3d[block_num];
	Point3d * max_points = new Point3d[block_num];
	Point3d * min_points_device;
	Point3d * max_points_device;

	if (min_points == nullptr) {
		std::cout << "memory allocating for id2boundingbox failed\n";
		return 0;
	}

	int index;
	for (int id = 0; id < block_num; id++) {
		bounding_box_file >> min_points[id].x;
		bounding_box_file >> min_points[id].y;
		bounding_box_file >> min_points[id].z;
		bounding_box_file >> max_points[id].x;
		bounding_box_file >> max_points[id].y;
		bounding_box_file >> max_points[id].z;
		//for (int i = 0; i < 6; i++){
		//	bounding_box_file >> a;
		//}
		bounding_box_file >> index;
	}

	CUDA_CALL(cudaMalloc(&min_points_device, sizeof(Point3d)*block_num));
	CUDA_CALL(cudaMemcpy(min_points_device, min_points, sizeof(Point3d)*block_num, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&max_points_device, sizeof(Point3d)*block_num));
	CUDA_CALL(cudaMemcpy(max_points_device, max_points, sizeof(Point3d)*block_num, cudaMemcpyHostToDevice));
	delete[] min_points;
	delete[] max_points;
	total_size = width * depth * height;

	std::cout << data_source << " " << integration_cluster_address << " " << sgmm_binary_address << " " << width << " " << depth << " " << height << std::endl;
	std::cout << "Part0: Initializing..." << std::endl;

	unsigned char * raw_result_host = (unsigned char *)malloc(total_size * sizeof(unsigned char));

	double * temp_p_host = (double *)malloc(total_size * sizeof(double));
	if (temp_p_host == nullptr) {
		std::cout << "error" << __LINE__ << std::endl;
		return 0;
	}

	CUDA_CALL(cudaMalloc(&raw_result, total_size * sizeof(unsigned char)));
	CUDA_CALL(cudaMalloc(&all_block_integrations, block_num * sizeof(sgmmClusterIntegrations)));
	CUDA_CALL(cudaMalloc(&temp_p, total_size * sizeof(double)));


#if defined(DEBUG_RESTOREVOXEL_KERNEL)||defined(DEBUG_DENOMINATOR_KERNEL)||defined(DEBUG_NUMERATOR_KERNEL)
	Point3d * debug_point1_host = new Point3d[block_num];
	Point3d * debug_point2_host = new Point3d[block_num];
	debug_block_info * debug_block_info_host = new debug_block_info[block_num];
	debug_block_info * debug_block_info_device;
	for (int i = 0; i < block_num; i++) {
		new(debug_point1_host + i)Point3d(44, 44, 44);
		new(debug_point2_host + i)Point3d(44, 44, 44);
	}
	CUDA_CALL(cudaMalloc(&debug_points1_device, block_num * sizeof(Point3d)));
	CUDA_CALL(cudaMalloc(&debug_points2_device, block_num * sizeof(Point3d)));
	CUDA_CALL(cudaMalloc(&debug_block_info_device, block_num * sizeof(debug_block_info)));

	CUDA_CALL(cudaMemcpy(debug_block_info_device, debug_block_info_host, block_num * sizeof(debug_block_info), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(debug_points1_device, debug_point1_host, block_num * sizeof(Point3d), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(debug_points2_device, debug_point2_host, block_num * sizeof(Point3d), cudaMemcpyHostToDevice));
#endif

	for (int i = 0; i < total_size; i++) {
		raw_result_host[i] = 125; //初始化为一个醒目的中间值
		temp_p_host[i] = 1.000;
	}
	CUDA_CALL(cudaMemcpy(raw_result, raw_result_host, total_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(temp_p, temp_p_host, total_size * sizeof(double), cudaMemcpyHostToDevice));

	// Part1: 读取SGMM数据
	sgmmClusterBlock* block_data;
	//cudaMallocManaged(&block_data, block_num * sizeof(Block));
	block_data = (sgmmClusterBlock*)malloc(sizeof(sgmmClusterBlock)*block_num);
	std::cout << std::endl << "Part1: Reading SGMMs..." << std::endl;
	std::ifstream f_sgmm(sgmm_binary_address, std::ios::binary);
	if (f_sgmm.is_open() == false) {
		std::cout << "sgmm file doesnt exist\n";
		return 0;
	}
	for (int block_index = 0; block_index < block_num; block_index++) {
		f_sgmm.read((char*)&(block_data[block_index].cluster_num_), sizeof(unsigned char));

		for (int cluster_index = 0; cluster_index < block_data[block_index].cluster_num_; cluster_index++) {
			f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].probability_), sizeof(float));
			f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gauss_count_), sizeof(unsigned char));
			f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].sample_value_), sizeof(float));
			for (int gauss_index = 0; gauss_index < block_data[block_index].clusters_[cluster_index].gauss_count_; gauss_index++) {
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].weight_), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].determinant_), sizeof(float));
				for (int i = 0; i < 3; i++) {
					f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].mean_[i]), sizeof(float));
				}
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[0]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[1]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[2]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[4]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[5]), sizeof(float));
				f_sgmm.read((char*)&(block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[8]), sizeof(float));
				block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[3] = block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[1];
				block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[6] = block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[2];
				block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[7] = block_data[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[5];
			}
		}
	}
	f_sgmm.close();
	sgmmClusterBlock * block_data_device;
	CUDA_CALL(cudaMalloc(&block_data_device, block_num*(sizeof(sgmmClusterBlock))));
	CUDA_CALL(cudaMemcpy(block_data_device, block_data, block_num*(sizeof(sgmmClusterBlock)), cudaMemcpyHostToDevice));
	free(block_data);
	block_data = block_data_device;


	//Part2: 读取每个点与id的对应关系

	//读取coord2id文件
	std::ifstream id_table_file(disk_address + data_source + std::string{ ".reidt" });
	if (id_table_file.is_open() == false) {
		std::cout << "can not open coord2id file\n";
		return 0;
	}
	id_type *id_table = new id_type[total_size];
	id_type * id_table_device;
	if (id_table == nullptr) {
		std::cout << "memory allocating for coord2id failed\n";
		return 0;
	}
	for (int i = 0; i < total_size; i++) {
		id_table_file >> id_table[i];
	}
	//id_table_file.read((char *)id_table, sizeof(id_type)*total_size);
	id_table_file.close();
	CUDA_CALL(cudaMalloc(&id_table_device, sizeof(id_type)*total_size));
	CUDA_CALL(cudaMemcpy(id_table_device, id_table, sizeof(id_type)*total_size, cudaMemcpyHostToDevice));
	delete[] id_table;

	// Part3~5: 获取SGMM的积分
	int numBlocks;
	clock_t start, finish;
	int read_from;
	std::cout << std::endl << "Choose a way to get integrations, input 1 to calc or 2 to read." << std::endl;
	std::cin >> read_from;
	sgmmClusterIntegrations * all_block_integrations_host = (sgmmClusterIntegrations *)malloc(block_num * sizeof(sgmmClusterIntegrations));
	if (read_from == 1) {
		// Part2~4: 计算积分
		// Part2: 计算所有block内的每一个cluster的sgmm在块内的积分的分子部分
		std::cout << std::endl << "Part2: Calculating integrations numerator..." << std::endl;
		numBlocks = (block_num * MAX_CLUSTER_NUM + blockSize - 1) / blockSize;
		start = clock();



#ifdef MONTE_CARLO

		MonteCarloIntegrations<< < numBlocks, blockSize >> > (all_block_integrations,
			block_data,
			n,
			block_num,
			MAX_CLUSTER_NUM,
			temp_p,
			temp_p2,
			debug_points1_device,
			debug_points2_device,
			min_points_device,
			max_points_device);;
#else

		CalcIntegrationsNumerator << < numBlocks, blockSize >> > (all_block_integrations,
			block_data,
			n,
			block_num,
			MAX_CLUSTER_NUM,
			temp_p,
			temp_p2,
			debug_points1_device,
			debug_points2_device,
			min_points_device,
			max_points_device);
#endif

		CUDA_CALL(cudaDeviceSynchronize());

		finish = clock();
		std::cout << "Calculating numerator time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "s" << std::endl;


		// Part3: 计算所有block内的每一个cluster的sgmm在块内的积分的分母部分,需要计算integration_scale*integration_scale*integration_scale个小块

#ifndef MONTE_CARLO
		std::cout << std::endl << "Part3: Calculating integrations denominator..." << std::endl;

		//// 防卡死措施，从文件读取目前的计算结果
		int start_index = 0;
		//std::ifstream f_temp_integration_in("d:/i_" + Int2String(start_index-1), std::ios::binary);
		//f_temp_integration_in.read((char *)all_block_integrations, block_num * sizeof(Integrations));
		//f_temp_integration_in.close();


		for (int loop_index = start_index; loop_index < integration_scale*integration_scale*integration_scale; loop_index++) {
			std::cout << "Calculating block " << loop_index << "..." << std::endl;
			start = clock();
			CalcIntegrationsDenominator << < numBlocks, blockSize >> > (all_block_integrations, block_data,
				n,
				block_num,
				MAX_CLUSTER_NUM,
				loop_index,
				integration_scale,
				temp_p,
				temp_p2,
				debug_points1_device,
				debug_points2_device,
				min_points_device,
				max_points_device);
			CUDA_CALL(cudaDeviceSynchronize());
			finish = clock();
			std::cout << "Calculating denominator time: " << 1.0 * (finish - start) / CLOCKS_PER_SEC << "s" << std::endl;

			// 防卡死措施，存储目前的计算结果到文件中
			//Sleep(1000);
			CUDA_CALL(cudaMemcpy(all_block_integrations_host, all_block_integrations, block_num*(sizeof(sgmmClusterIntegrations)), cudaMemcpyDeviceToHost));

			std::ofstream f_temp_integration_out("d:/i_" + Int2String(loop_index), std::ios::binary);

			f_temp_integration_out.write((char *)all_block_integrations_host, block_num * sizeof(sgmmClusterIntegrations));
			f_temp_integration_out.close();

			//Mark(Int2String(loop_index));
		}
#endif


		// Part4: 存储积分到一个单独的文件
		std::cout << std::endl << "Part4: Saving integrations..." << std::endl;

		std::ofstream f_out(integration_cluster_address, std::ios::binary);
		for (int i = 0; i < block_num; i++) {
			for (int j = 0; j < MAX_CLUSTER_NUM; j++) {
				f_out.write((char*)&(all_block_integrations_host[i].integration_value[j]), sizeof(float));
			}
		}

		std::ofstream integration_out_text(integration_cluster_address+"_txt");
		std::cout << "Wrting integraion file as text format\n";
		if (integration_out_text.is_open() == true) {
			for (int i = 0; i < block_num; i++) {
				integration_out_text << "------------block num:" << i << std::endl;
				for (int j = 0; j < MAX_CLUSTER_NUM; j++) {
					integration_out_text << all_block_integrations_host[i].integration_value[j] << std::endl;
				}
			}
		}
		else {
			std::cout << "can not create text file for integrations\n";
		}

	}
	else if (read_from == 2) {
		// Part2~4: 从文件中读取积分
		std::cout << std::endl << "Part2~4: Reading integrations..." << std::endl;
		std::ifstream f_in(integration_cluster_address, std::ios::binary);
		for (int i = 0; i < block_num; i++) {
			for (int j = 0; j < MAX_CLUSTER_NUM; j++) {
				f_in.read((char*)&(all_block_integrations_host[i].integration_value[j]), sizeof(float));
			}
		}

		CUDA_CALL(cudaMemcpy(all_block_integrations, all_block_integrations_host, block_num * sizeof(sgmmClusterIntegrations), cudaMemcpyHostToDevice));
	}
	else {
		exit(1);
	}
	free(all_block_integrations_host);


	// Part5: 重构raw数据
	std::cout << std::endl << "Part5: Restoring data..." << std::endl;
	std::cout << std::endl << "Choose a way to restore data, input 1 to sample randomly, 2 to sample by max, 3 to sample by integration, 4 to sample by rejection method." << std::endl;
	int sample_choice;
	std::cin >> sample_choice;
	if (sample_choice != 1 && sample_choice != 2 && sample_choice != 3 && sample_choice != 4) {
		std::cout << "Illegal input." << std::endl;
		exit(1);
	}
	//numBlocks = (((width / side)*side) * ((depth / side)*side) * ((height / side)*side) + blockSize - 1) / blockSize; //需要保证三维是side的倍数
	numBlocks = (total_size + blockSize - 1) / blockSize;
	start = clock();
	//RestoreVoxel << <numBlocks-1, blockSize >> >(raw_result, width, depth, height, side, n, MAX_CLUSTER_NUM, width_num, depth_num, height_num, block_data,
	//	temp_p, temp_p2, all_block_integrations, sample_choice);
	RestoreVoxel << <numBlocks, blockSize >> > (raw_result,
		width,
		depth,
		height,
		n, 
		MAX_CLUSTER_NUM, 
		block_data,
		temp_p,
		temp_p2,
		all_block_integrations, 
		sample_choice,
		id_table_device, 
		min_points_device,
		max_points_device,
		debug_block_info_device,
		total_size);
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

	//写回temp数组
	// Part6: 还原为raw文件
	std::cout << std::endl << "Part6: Saving raw file..." << std::endl;
	std::ofstream f_result(result_name, std::ios::binary);
	CUDA_CALL(cudaMemcpy(raw_result_host, raw_result, total_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	f_result.write((const char*)raw_result_host, total_size); //注意：多余的数据用0填充了
	f_result.close();
	cudaFree(raw_result);
	cudaFree(temp_p);
	cudaFree(block_data);
	cudaFree(all_block_integrations);
	cudaFree(id_table_device);
	cudaFree(min_points_device);
	cudaFree(max_points_device);

	//DEBUG:writing debug info
#ifdef DEBUG_RESTOREVOXEL_KERNEL
	//CUDA_CALL(cudaMemcpy(temp_p_host, temp_p, total_size * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(debug_block_info_host, debug_block_info_device, block_num * sizeof(debug_block_info), cudaMemcpyDeviceToHost));
	std::ofstream out_debug_file(disk_address + data_source + ".dbg");
	std::ofstream out_debug_info_file(disk_address + data_source + ".dbgblockinfo");
	if (out_debug_file.is_open() == false) {
		std::cout << "can not open .dbg file\n";
	}
	//for (int i = 0; i < total_size; i++) {
	//	out_debug_file << (int)temp_p_host[i] << std::endl;
	//}
	out_debug_file.close();
	if (out_debug_info_file.is_open() == false) {
		std::cout << " can not open .dbgblockinfo\n";
	}
	for (int i = 0; i < block_num; i++) {
		out_debug_info_file << debug_block_info_host[i].min_point << " " << debug_block_info_host[i].max_point << " " << debug_block_info_host[i].cluster_num;
		int cluster_num = debug_block_info_host[i].cluster_num;
		for (int j = 0; j < cluster_num; j++) {
			out_debug_info_file << " " << debug_block_info_host[i].gauss_count[j];
		}
		out_debug_info_file << std::endl;
	}
	
	out_debug_info_file.close();
	delete[] temp_p_host;
#endif

#ifdef DEBUG_NUMERATOR_KERNEL
	CUDA_CALL(cudaMemcpy(debug_point1_host, debug_points1_device, block_num * sizeof(Point3d), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(debug_point2_host, debug_points2_device, block_num * sizeof(Point3d), cudaMemcpyDeviceToHost));
	std::ofstream out_dbgnm_file(disk_address + data_source + ".dbgnm");
	if (out_dbgnm_file.is_open() == false) {
		std::cout << "can not open .dbgnm file\n";
	}
	for (int i = 0; i < block_num; i++) {
		out_dbgnm_file << debug_point1_host[i] << " " << debug_point2_host[i] << " " << i << std::endl;
	}
#endif

#ifdef DEBUG_DENOMINATOR_KERNEL
	CUDA_CALL(cudaMemcpy(debug_point1_host, debug_points1_device, block_num * sizeof(Point3d), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(debug_point2_host, debug_points2_device, block_num * sizeof(Point3d), cudaMemcpyDeviceToHost));
	std::ofstream out_dbgdnm_file(disk_address + data_source + ".dbgdnm");
	if (out_dbgdnm_file.is_open() == false) {
		std::cout << "can not open .dbgdnm file\n";
	}
	for (int i = 0; i < block_num; i++) {
		out_dbgdnm_file << debug_point1_host[i] << " " << debug_point2_host[i] << " " << i << std::endl;
	}
#endif


#if defined(DEBUG_RESTOREVOXEL_KERNEL)||defined(DEBUG_DENOMINATOR_KERNEL)||defined(DEBUG_NUMERATOR_KERNEL)
	delete[] debug_point1_host;
	delete[] debug_point2_host;
	delete[] debug_block_info_host;
	cudaFree(debug_points1_device);
	cudaFree(debug_points2_device);
	cudaFree(debug_block_info_device);
#endif

	//Creating .vifo file
	create_vifo_file(disk_address, data_source + "_restored_sgmm_cluster", width, depth, height);

	std::cin.get();
	return 0;
}
