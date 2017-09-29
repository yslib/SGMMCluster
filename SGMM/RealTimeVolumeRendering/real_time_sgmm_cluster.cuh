#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>


#define MAX_CLUSTER_NUM 10 // 最大聚类数
#define MAX_GAUSS_COMPONENT_NUM 4 // 最大高斯分支数
#define M_PI 3.14159265358979323846

using id_type = int;
using pos_type = unsigned int;

struct Gauss {
	float weight_;
	float mean_[3];
	float precisions_[9];
	float determinant_;
	Gauss() :weight_{ 0.0 }, determinant_{ 1.0 }{
		for (int i = 0; i < 3; i++){
			mean_[i] = 0.0;
		}
		for (int i = 0; i < 9; i++){
			precisions_[i] = 0.0;
		}
	}
};

struct Cluster {
	float probability_ = 0.0;
	unsigned char gauss_count_ = 0;
	float sample_value_ = 0;
	Gauss gausses_[MAX_GAUSS_COMPONENT_NUM];
};


struct Block {
	int index_ = 0;
	unsigned char cluster_num; // 实际聚类的数量
	Cluster clusters_[MAX_CLUSTER_NUM];
};
struct Integrations {
	float integration_numerator[10];
	float integration_denominator[10];
	float integration_value[10];
};

struct point3d{
	pos_type x;
	pos_type y;
	pos_type z;
	point3d(pos_type xx = 0, pos_type yy = 0, pos_type zz = 0) :
		x{ xx }, y{ yy }, z{ zz }{}
	point3d operator+(const point3d & p)const{
		return point3d{ x + p.x, y + p.y, z + p.z };
	}
};


void readResource();

__device__
int CalcSampleValueSGMMCluster(Block* block_data_device,
	Integrations* all_block_integrations_device, 
	float global_width_pos,
	float global_depth_pos,
	float global_height_pos,
	int n, 
	int sample_choice, 
	int calc_index, 
	int side, 
	int max_cluster_num, 
	int width_num,
	int depth_num);


__device__
int ReadValueFromSrcRaw(unsigned char* raw_src_device, 
	float global_width_pos, 
	float global_depth_pos, 
	float global_height_pos, 
	int real_width,
	int real_depth, 
	int real_height);

extern Integrations * all_block_integrations_device;
extern Integrations * all_block_integrations_host;
extern float * data_trans_device;
extern float * data_trans_host;
extern Block * block_data_host;
extern Block * block_data_device;

extern point3d * min_points;
extern point3d * min_points_device;

extern id_type * block_id_device;
extern id_type * block_id_host;
extern int side;

extern int real_width;
extern int real_depth;
extern int real_height;

extern int block_width;
extern int block_depth;
extern int block_height;
extern int block_size;

extern int max_cluster_num;
extern int width_num;
extern int depth_num;
extern int height_num;
extern int block_num;
extern int sample_choice;
extern int n;
extern unsigned char * raw_src_host;
extern unsigned char * raw_src_device;

// 检查光线是否在缩放后的物体内
