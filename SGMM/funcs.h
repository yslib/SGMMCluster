#pragma once
#ifndef _STRUCTURESFUNCTIONS_H_
#define _STRUCTURESFUNCTIONS_H_
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>

#define MAX_CLUSTER_NUM 10 // 最大聚类数
#define MAX_GAUSS_COMPONENT_NUM 4 // 最大高斯分支数


#define ERROR_INFO
#ifdef ERROR_INFO
#define CUDA_CALL(x) {const cudaError_t a =(x);\
if(a != cudaSuccess){printf("\nCUdaError:%s(line:%d)\n",\
cudaGetErrorString(a),__LINE__);cudaDeviceReset();}}
#else
#define CUDA_CALL(x) (x)
#endif

using voxel_type = unsigned char;
using pos_type = int;
using id_type = int;


enum class data_type { BLOCL_GMM, SGMM, SGMM_CLUSTER };

struct point3d {
	pos_type x;
	pos_type y;
	pos_type z;
	point3d(pos_type xx = 0, pos_type yy = 0, pos_type zz = 0) :
		x{ xx }, y{ yy }, z{ zz } {}
	point3d operator+(const point3d & p)const {
		return point3d{ x + p.x, y + p.y, z + p.z };
	}
	friend std::ostream & operator<<(std::ostream & os, const point3d & p) {
		os << p.x << " " << p.y << " " << p.z;
		return os;
	}
	friend std::istream & operator>>(std::istream & is, point3d & p) {
		is >> p.x >> p.y >> p.z;
		return is;
	}
	bool operator<(const point3d & p)const {
		return (x < p.x&&y < p.y&&z < p.z);
	}
	bool operator<=(const point3d & p)const {
		return (x <= p.x&&y <= p.y&&z <= p.z);
	}
	bool operator>=(const point3d & p)const {
		return !(*this < p);
	}
	bool operator>(const point3d & p)const {
		return !(*this <= p);
	}
	bool operator==(const point3d & p)const {
		return (x == p.x&&y == p.y&&z == p.z);
	}
	bool operator!=(const point3d & p)const {
		return !(*this == p);
	}

};

struct AABB{
	AABB(const point3d & p1 = point3d(), const point3d & p2 = point3d()) :min_point{ p1 }, max_point{ p2 } {}
	point3d min_point;
	point3d max_point;
};

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
	int index_ = 0;
	unsigned char cluster_num_; // 实际聚类的数量
	Cluster clusters_[MAX_CLUSTER_NUM];
};


struct sgmmClusterIntegrations {
	float integration_numerator[MAX_CLUSTER_NUM];
	float integration_denominator[MAX_CLUSTER_NUM];
	float integration_value[MAX_CLUSTER_NUM];
};

std::size_t get_file_size(const std::string & path);

std::string int_to_string(int i);

bool read_raw_file(const std::string & file_name, voxel_type * vol, size_t width, size_t depth, size_t height);

bool create_vifo_file(const std::string & address,const std::string & file_name, int width, int depth, int height);

std::vector<AABB> create_regular_boundingbox(int width,int depth,int height,int block_width,int block_depth,int block_height);

bool create_AABB_file(const std::string & path, const std::string & file_name, const std::vector<AABB> & aabbs);

std::vector<AABB> read_AABB_from_file(const std::string path, const std::string & file_name);

#endif