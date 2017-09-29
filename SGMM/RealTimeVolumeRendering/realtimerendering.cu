#include <cstdio>
#include <cstdlib>
#include "realtimerendering.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include "real_time_sgmm_cluster.cuh"
#include <string>
#include <fstream>
#include <iostream>


#include "interaction.h"
//#define SQUARE


#define ERROR_INFO

#ifdef ERROR_INFO
#define CUDA_CALL(x) {const cudaError_t a =(x);\
if(a != cudaSuccess){printf("\nCUdaError:%s(line:%d)\n",\
cudaGetErrorString(a),__LINE__);cudaDeviceReset();}}
#else
#define CUDA_CALL(x) (x)
#endif


//const int N = 64;
//const int TPB = 32;
//const int TX = 8;
//const int TY = 8;
//const int TZ = 8;
//
const int TX_2D = 32;
const int TY_2D = 32;
//
//const int RAD = 1;

__device__ float distance(float x1,float x2) {
	return sqrt((x1 - x2)*(x1 - x2));
}
__device__ unsigned char clip(int n) {
	return n > 255?255 : (n < 0 ? 0 : n);
}
// Atomic operation

//__global__
//void dotKernel(int *d_res, const int *d_a, const int *d_b, int n) {
//	const int idx = blockDim.x*blockIdx.x + threadIdx.x;
//	if (idx >= n)return;
//	const int s_idx = threadIdx.x;
//
//	__shared__ int s_prob[TPB];
//	s_prob[threadIdx.x] = d_a[idx] * d_b[idx];
//	__syncthreads();
//	if (threadIdx.x == 0) {
//		int blockSum = 0;
//		for (int i = 0; i < blockDim.x;i++) {
//			blockSum += s_prob[i];
//		}
//		atomicAdd(d_res, blockSum);// acessing a shared varible
//	}
//	
//}
//__global__ void distanceKernel(uchar4 * d_out, int w, int h, int2 pos) {
//	const int c = blockDim.x*blockIdx.x + threadIdx.x;
//	const int r = blockDim.y*blockIdx.y + threadIdx.y;
//	const int i = r*w + c;
//	if (r >= h || c >= w)
//		return;
//	int d =  sqrtf((c-pos.x)*(c-pos.x) + (r-pos.y)*(r-pos.y));
//	const unsigned char intensity = clip(255 - d);
//	d_out[i].x = intensity;
//	d_out[i].y = intensity;
//	d_out[i].z = 0;
//	d_out[i].w = 255;
//}
//void kernelLauncher(uchar4 * d_out, int w, int h, int2 pos) {
//	const dim3 blockSize(TX, TY);
//	const dim3 gridSize((w+TX-1)/TX,(h+TY-1)/TY);
//	distanceKernel <<<gridSize, blockSize >> > (d_out, w, h, pos);
//}

//
//Shared Memory

//
//__global__ void ddKernel(float * d_out, const float *d_in, int size, int h) {
//	const int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if (i >= size)return;
//	const int s_idx = threadIdx.x + RAD;
//	extern __shared__ float s_in[];
//	//Or
//	//__shared__ float s_in[blockDim+2*RAD];
//	//
//	s_in[s_idx] = d_in[i];
//	if (threadIdx.x < RAD) {
//		s_in[s_idx - RAD] = d_in[i - RAD];
//		s_in[s_idx + blockDim.x] = d_in[i + blockDim.x];
//	}
//	__syncthreads();
//
//	d_out[i] = (s_in[s_idx - 1] - 2.f*s_in[s_idx] + s_in[s_idx + 1]) / (h*h);
//
//}

//__global__ void centroidKernel(const uchar4 * d_img, int * d_centroidCol, int * d_centroidRow, int *d_pixelCount
//	,int width, int height) {
//	__shared__ uint4 s_img[TPB];
//
//	const int idx = blockDim.x*blockIdx.x + threadIdx.x;
//	const int s_idx = threadIdx.x;
//	const int row = idx / width;
//	const int col = idx - row*width;
//
//	if ((d_img[idx].x < 255 || d_img[idx].y<255 || d_img[idx].z < 255) && (idx<width*height)) {
//		s_img[s_idx].x = col;
//		s_img[s_idx].y = row;
//		s_img[s_idx].z = 1;
//	}
//	else {
//		s_img[s_idx].x = 0;
//		s_img[s_idx].y = 0;
//		s_img[s_idx].z = 0;
//	}
//	__syncthreads();
//	for (int s = blockDim.x / 2; s >= 1; s >>= 1) {
//		if (s_idx < s) {
//			s_img[s_idx] += s_img[s_idx + s];
//		}
//		__syncthreads();
//	}
//	
//	
//}

///////////////////////////////////////////////
///////////////////////////////////////////////
///////////////////////////////////////////////
struct Ray {
	float3 o, d;
};
__device__ float3 paramRay(Ray r, float t) {
	return r.o + t*r.d;
}
__device__ int3 posToIndex(float3 pos, int3 volSize) {
	return make_int3(pos.x + volSize.x / 2, pos.y + volSize.y / 2, pos.z + volSize.z / 2);
}
__device__ int flatten(int3 index, int3 volSize) {
	return index.x + index.y*volSize.x + index.z*volSize.y*volSize.x;
}
__device__ int clipWithBounds(int n, int n_min, int n_max) {
	return n < n_min ? n_min : (n > n_max ? n_max : n);
}
__device__ float panelSDF(float3 pos,float3 norm,float d) {
	return dot(pos, normalize(norm)) - d;
}
__device__ float3 scrIdxToPos(int c, int r, int w, int h, float zs) {
	return make_float3(c - w / 2, r - h / 2, zs);
}
__device__ bool rayPlaneIntersect(Ray ray,float3 norm,float d,float * t) {
	float f1 = panelSDF(paramRay(ray,0.f), norm, d);
	float f2 = panelSDF(paramRay(ray,1.0f), norm, d);
	bool intersect = (f1*f2 < 0);
	if (intersect)
		*t = (.0f-f1 )/ (f2 - f1);
	return intersect;
}
__device__ float3 yRotate(float3 pos, float theta) {
	const float c = cosf(theta), s = sinf(theta);
	return make_float3(c*pos.x + s*pos.z, pos.y, -s*pos.x + c*pos.z);
}

__device__ float density(float * d_vol, int3 volSize, float3 pos) {
	int3 index = posToIndex(pos, volSize);
	const int w = volSize.x, h = volSize.y, d = volSize.z;
	int i = index.x, j = index.y, k = index.z;
	const float3 rem = fracf(pos);
	index = make_int3(clipWithBounds(i, 0, w-2), clipWithBounds(j, 0, h - 2), clipWithBounds(k, 0, d - 2));

	const int3 dx = { 1,0,0 }, dy = { 0,1,0 }, dz = { 0,0,1 };
	const float dens000 = d_vol[flatten(index,volSize)];
	const float dens001 = d_vol[flatten(index + dz, volSize)];
	const float dens010 = d_vol[flatten(index + dy, volSize)];
	const float dens011 = d_vol[flatten(index + dz+dy, volSize)];
	const float dens100 = d_vol[flatten(index + dx, volSize)];
	const float dens101 = d_vol[flatten(index + dx + dz, volSize)];
	const float dens110 = d_vol[flatten(index + dx + dy,volSize)];
	const float dens111 = d_vol[flatten(index+dx+dy+dz,volSize)];
	float res = 0.f;
	res = (1 - rem.x)*(1 - rem.y)*(1 - rem.z)*dens000 +
		(rem.x)*(1 - rem.y)*(1 - rem.z)*dens100 +
		(1 - rem.x)*(rem.y)*(1 - rem.z)*dens010 +
		(1 - rem.x)*(1-rem.y)*(rem.z)*dens001 +
		(1 - rem.x)*(rem.y)*(rem.z)*dens011 +
		(rem.x)*(rem.y)*(1-rem.z)*dens110 +
		(rem.x)*(1-rem.y)*(rem.z)*dens101 +
		(rem.x)*(rem.y)*(rem.z)*dens111;
	//printf("%.2lf\n", res);
	return res;
}
__device__ float func(int c, int r, int s, int id, int3 volSize, float4 params) {
	const int3 pos0 = { volSize.x / 2,volSize.y / 2,volSize.z / 2 };
	const float dx = c - pos0.x, dy = r - pos0.y, dz = s - pos0.z;
	if (id == 0)return sqrtf(dx*dx + dy*dy + dz*dz) - params.x;
	else if (id == 1) {
		const float r = sqrtf(dx*dx + dy*dy);
		return sqrtf((r-params.x)*(r-params.x) + dz*dz) - params.y;
	}
	else {
		float x = fabsf(dx) - params.x, y = fabsf(dy) - params.y,
			z = fabsf(dz) - params.z;
		if (x <= 0 && y <= 0 && z <= 0)return fmaxf(x, fmaxf(y, z));
		else {
			x = fmaxf(x, 0), y = fmaxf(y, 0), z = fmaxf(z, 0);
			return sqrt(x*x + y*y + z*z);
		}
	}
}

__device__ const float  EPS= 0.01f;
const int NUMSTEPS = 20;

//
//intersection of axis-align bounding box and ray 
//
__device__ bool intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar) {
	const float3 invR = make_float3(1.0f) / r.d;
	const float3 tbot = invR*(boxmin - r.o), ttop = invR*(boxmax - r.o);
	const float3 tmin = fminf(ttop, tbot), tmax = fmaxf(ttop, tbot);

	*tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	*tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
	return *tfar > *tnear;
}
__device__ uchar4 sliceShader(float *d_vol, int3 volSize, Ray boxRay, float gain, float d, float3 norm) {
	float t;
	uchar4 shade = make_uchar4(96, 0, 192, 0);//Backgound color
	if (rayPlaneIntersect(boxRay, norm, d, &t)) {

		float den = density(d_vol, volSize, paramRay(boxRay, t));
		shade = make_uchar4(48, clip(-10.f*(1.0f + gain)*den), 96, 255);
	}
	return shade;
}
__device__ uchar4 volumeRenderShader(float *d_vol, int3 volSize, Ray boxRay, float threshold,int numSteps) {
	uchar4 shade = make_uchar4(96, 0, 192, 0);//Backgound colorr
	const float dt = 1.f / numSteps;
	const float len = length(boxRay.d) / numSteps;
	float accum = 0.0f;
	float3 pos = boxRay.o;
	float val = density(d_vol, volSize, pos);
	for (float t = 0.0f; t < 1.0f; t += dt) {
		if (val - threshold < 0.f)
			accum += (fabsf(val - threshold))*len;
		pos = paramRay(boxRay, t);
		val = density(d_vol, volSize, pos);
	}
	if (clip(accum) > 0.f)shade.y = clip(accum);
	return shade;
}
__device__ uchar4 rayCastShader(float *d_vol, int3 volSize, Ray boxRay, float d) {
	uchar4 shade = make_uchar4(96, 0, 192, 0);//Backgound color
	float3 pos = boxRay.o;
	float len = length(boxRay.d);
	float t = 0.0f;
	float f = density(d_vol, volSize, pos);
	while (f > d + EPS &&t < 1.0f) {
		f = density(d_vol, volSize, pos);
		t += (f-d) / len;
		pos = paramRay(boxRay, t);
		f = density(d_vol, volSize, pos);
	}
	if (t < 1.f) {
		const float3 ux = make_float3(1, 0, 0), uy = make_float3(0, 1, 0), uz = make_float3(0, 0, 1);
		float3 grad = { (density(d_vol,volSize,pos + EPS*ux) - density(d_vol,volSize,pos)) / EPS,
			(density(d_vol,volSize,pos + EPS*uy) - density(d_vol,volSize,pos)) / EPS,
			(density(d_vol,volSize,pos + EPS*uz) - density(d_vol,volSize,pos)) / EPS
		};
		float intensity = -dot(normalize(boxRay.d), normalize(grad));
		shade = make_uchar4(255 * intensity, 0, 0, 255);
	}
	return shade;
}


static Integrations * all_block_integrations_device;
static Integrations * all_block_integrations_host;

static Block * block_data_host;
static Block * block_data_device;

static point3d *min_points;
static point3d *min_points_device;

static float * data_trans_device;
static float * data_trans_host;

static id_type * block_id_host;
static id_type * block_id_device;

static int real_width;
static int real_depth;
static int real_height;

static int side;
static int max_cluster_num;
static int block_width;
static int block_depth;
static int block_height;
static int block_size;
static int width_num;
static int depth_num;
static int height_num;
static int block_num;
static int sample_choice;
static int n;
static unsigned char * raw_src_host;
static unsigned char * raw_src_device;


__device__ float exp_table_device[100000];
__device__ float lower_device;
__device__ float upper_device;
__device__ int count_device;
__device__ float step_device;


__global__
void calcExpKernel() {
	int index = blockDim.x*blockIdx.x + threadIdx.x;

	if (index >= count_device)return;
	//printf("%d %d:\n", index, count_device);
	exp_table_device[index] = __expf(lower_device + index*step_device);
	//printf("%d: exp(%f) = %f\n", index, lower_device + index*step_device, exp_table_device[index]);
}


void calcExp(float step_host, float lower_host, float upper_host) {
	int count_host = int((upper_host - lower_host) / step_host);
	//float *dptr_step=nullptr, *dptr_lower, *dptr_upper, *dptr_count;
	//CUDA_CALL(cudaGetSymbolAddress((void **)dptr_step, &step_device));
	CUDA_CALL(cudaMemcpyToSymbol(step_device, &step_host, sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(lower_device, &lower_host, sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(upper_device, &upper_host, sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(count_device, &count_host, sizeof(int)));

	int blockSize = 32;
	int gridSize = (count_host + blockSize - 1) / blockSize;
	printf("%f %f %f %d %d %d\n", step_host, lower_host, upper_host, count_host, gridSize, blockSize);

	calcExpKernel << <gridSize, blockSize >> >();
	CUDA_CALL(cudaDeviceSynchronize());

}

__device__ float exp_table(float x) {
	if (x < lower_device)return 0.0f;
	else if (x > upper_device)return exp_table_device[count_device - 1];
	return exp_table_device[int((x - lower_device) / step_device)];
}



// 计算A^T * B * C
__device__
static double MulMatrix(double * A, float * B, double * C) {
	return (A[0] * B[0] + A[1] * B[3] + A[2] * B[6]) * C[0] + (A[0] * B[1] + A[1] * B[4] + A[2] * B[7]) * C[1] + (A[0] * B[2] + A[1] * B[5] + A[2] * B[8]) * C[2];
}


// 计算一个gmm分支
__device__
static double CalcGMM(float* mean, float* inv_cov, double determinant, double* local_pos, int n) {
	double diff[3]; //当前坐标和平均值的差值
	for (int i = 0; i < 3; i++) {
		diff[i] = local_pos[i] - mean[i];
	}
	return 1.0 / sqrt(8 * M_PI * M_PI * M_PI * determinant) * __expf(-0.5 * MulMatrix(diff, inv_cov, diff));
	//return 1.0 / sqrt(8 * M_PI * M_PI * M_PI * determinant) * exp_table(-0.5 * MulMatrix(diff, inv_cov, diff));
	//return 0;
}


// 计算local_pos位置处的sgmm（概率密度）
__device__
static double CalcSGMM(double* local_pos, int gauss_count, Block* block_data_device, int block_index, int cluster_index, int n) {
	double sgmm_result = 0.0;
	for (int gauss_index = 0; gauss_index < gauss_count; gauss_index++) {
		double added_value = block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].weight_
			* CalcGMM(block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].mean_,
				block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_,
				block_data_device[block_index].clusters_[cluster_index].gausses_[gauss_index].determinant_,
				local_pos, n);
		sgmm_result += added_value;
	}
	return sgmm_result;
}


// 计算采样值
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
int depth_num,
int block_width,
int block_depth,
int block_height,
id_type * block_id_device,point3d * min_points) {
	// 计算块的索引

#ifdef SQUARE
	int height_index = (int)global_height_pos / side;
	int depth_index = (int)global_depth_pos / side;
	int width_index = (int)global_width_pos / side;
	int block_index = height_index * width_num * depth_num + depth_index * width_num + width_index;
#else
	int height_index = global_height_pos / block_height;
	int depth_index = global_depth_pos / block_depth;
	int width_index = global_width_pos / block_width;
	int id = height_index * width_num * depth_num + depth_index * width_num + width_index;
	//printf("%d %d %d %d\n", height_index,depth_index,width_index,id);
	int block_index = block_id_device[id];
	//printf("%d %d\n", id,block_index);
	//int block_index = 0;
#endif
	
	//return 0;
	// 判断是不是全黑的空块
	if (block_data_device[block_index].cluster_num == 0) {
		return 0.0;
	}

	// 计算块内坐标
	double local_pos[3];
	local_pos[0] = (int)global_width_pos - min_points[block_index].x;  //x
	local_pos[1] = (int)global_depth_pos - min_points[block_index].y;  //y
	local_pos[2] = (int)global_height_pos - min_points[block_index].z; //z

	//printf("%f %f %f\n", local_pos[0], local_pos[1], local_pos[2]);
	//printf("%d\n", block_data_device);

																 // 计算所有cluster的概率
	double sgmmbi_l[MAX_CLUSTER_NUM];
	double P[MAX_CLUSTER_NUM];
	for (int cluster_index = 0; cluster_index < block_data_device[block_index].cluster_num; cluster_index++) {
		 //如果cluster的probability为0，则sgmmbi_i[i]为0，直接跳过
		if (block_data_device[block_index].clusters_[cluster_index].probability_ == 0) { // 这种情况不会发生
			sgmmbi_l[cluster_index] = 0.0;
			continue;
		}
		else {	// 计算SGMM（概率密度）
			sgmmbi_l[cluster_index] = CalcSGMM(local_pos, block_data_device[block_index].clusters_[cluster_index].gauss_count_, block_data_device, block_index, cluster_index, n);
		}
	}

	for (int cluster_index = 0; cluster_index < block_data_device[block_index].cluster_num; cluster_index++) {
		if (sgmmbi_l[cluster_index] == 0 || all_block_integrations_device[block_index].integration_value[cluster_index] == 0) {
			P[cluster_index] = 0.0;
		}
		else {
			P[cluster_index] = sgmmbi_l[cluster_index] * (block_data_device[block_index].clusters_[cluster_index].probability_ / all_block_integrations_device[block_index].integration_value[cluster_index]); // 除以sgmm在块内的积分
		}
	}

	// 归一化P   为什么一定要归一化
	double sum_p = 0.0;
	for (int i = 0; i < block_data_device[block_index].cluster_num; i++) {
		sum_p += P[i];
	}
	for (int i = 0; i < block_data_device[block_index].cluster_num; i++) {
		P[i] /= sum_p;
	}

	// 采样
	int	final_cluster_index = 0;
	if (sample_choice == 1) { // 方法一：随机采样
		curandState state;
		curand_init(calc_index, 100, 0, &state);
		double sample = curand_uniform(&state);
		double total_sum = 0.0;
		for (int i = 0; i < block_data_device[block_index].cluster_num; i++) {
			total_sum += P[i];
			if (total_sum > sample) {
				final_cluster_index = i;
				break;
			}
		}
	}
	else if (sample_choice == 2) { // 方法二：取概率最大采样
		double max_p = P[0];
		for (int i = 1; i < block_data_device[block_index].cluster_num; i++) {
			if (P[i] > max_p) {
				max_p = P[i];
				final_cluster_index = i;
			}
		}
	}
	else if (sample_choice == 3) { // 方法三：蒙特卡洛拒绝采样
		double max_p = P[0];
		final_cluster_index = 0;
		for (int i = 1; i < block_data_device[block_index].cluster_num; i++) {
			if (P[i] > max_p) {
				max_p = P[i];
				final_cluster_index = i;
			}
		}

		curandState state;
		curand_init(calc_index, 10, 0, &state);
		for (int i = 0; i<100; i++) {
			int sample_x = curand_uniform(&state) * block_data_device[block_index].cluster_num;
			double sample_y = curand_uniform(&state) * max_p;
			if (sample_y <= P[sample_x]) {
				final_cluster_index = sample_x;
				break;
			}
		}
	}

	//return 0;
	return block_data_device[block_index].clusters_[final_cluster_index].sample_value_;
}

__device__
int ReadValueFromSrcRaw(unsigned char* raw_src_device, float global_width_pos, float global_depth_pos, float global_height_pos, int real_width, int real_depth, int real_height) {
	int x = (int)(global_width_pos);
	int y = (int)(global_depth_pos);
	int z = (int)(global_height_pos);
	//return 0;
	return raw_src_device[z * real_width * real_depth + y * real_width + x];
}

__device__
uchar4 realTimeRenderForSGMMCluster(int3 vloSize,
Ray boxRay,
Block * block_data_device,
Integrations * all_block_integrations,
unsigned char * raw_src_device,
float * data_trans_device,
id_type * block_id_device,
int width,
int depth,
	int block_width,
	int block_depth,
	int block_height,
	point3d * min_points) 
{
	//printf("%d %d %d %d %d %d\n", width, depth, block_depth, block_width, block_height, block_id_device);
	uchar4 shade = make_uchar4(96, 0, 192, 0);//Backgound color
	float3 pos = boxRay.o;
	float dt = 1.0/length(boxRay.d);
	float t = 0.0f;
	float3 origin = -make_float3(vloSize.x/2, vloSize.y/2, vloSize.z/2);
	//printf("origin:%f %f %f\n", origin.x, origin.y, origin.z);
	float4 Cdst = { 0.0,0.0,0.0,0.0 };
	//printf("%d\n", block_data_device);
	while (t < 1.0) {
		float3 currentCoord = paramRay(boxRay, t);
		//printf("%f %f %f\n", currentCoord.x, currentCoord.y, currentCoord.z);
		float3 currentLocalCoord = currentCoord - origin;
		//printf("%f %f %f\n", currentLocalCoord.x, currentLocalCoord.y, currentLocalCoord.z);
		int sampleValue = 0;
		  sampleValue = CalcSampleValueSGMMCluster(block_data_device,
			all_block_integrations,
			currentLocalCoord.x, 
			currentLocalCoord.y,
			currentLocalCoord.z, 
			3, 
			3,			//unused
			0,			//unused
			16,			//unused
			MAX_CLUSTER_NUM,
			width/block_width, 
			depth/block_depth,block_width,block_depth,block_height,block_id_device,min_points);
		 

		//printf("%d %d %d\n", currentLocalCoord.x, currentLocalCoord.y, currentLocalCoord.z);
		//sampleValue = ReadValueFromSrcRaw(raw_src_device, 
		//	currentLocalCoord.x, 
		//	currentLocalCoord.y, 
		//	currentLocalCoord.z, 
		//	vloSize.x, 
		//	vloSize.y,
		//	vloSize.z);
		//printf("%d\n", sampleValue);
		
		//float4 Csrc;
		  float4 Csrc = { data_trans_device[4 * sampleValue],
			data_trans_device[4 * sampleValue + 1] ,
			data_trans_device[4 * sampleValue + 2] ,
			data_trans_device[4 * sampleValue + 3] };

		Cdst.x += (1 - Cdst.w)*Csrc.x*Csrc.w;
		Cdst.y += (1 - Cdst.w)*Csrc.y*Csrc.w;
		Cdst.z += (1 - Cdst.w)*Csrc.z*Csrc.w;
		Cdst.w += (1 - Cdst.w)*Csrc.w;

		t += dt;
	}
	//printf("%f %f %f %f\n", Cdst.x, Cdst.y, Cdst.z, Cdst.w);
	return make_uchar4(Cdst.x*255, Cdst.y*255, Cdst.z*255, Cdst.w*255);
}

__global__ void renderKernel(uchar4 *d_out, 
	float *d_vol, 
	int w,
	int h,
	int3 volSize,
	int method,
	int zs,
	float theta, 
	float threshold,
	float dist,
	Block * block_data_device,
	Integrations * all_block_integrations_device,
	unsigned char * raw_src_device,
	float * data_trans_device,
	id_type * block_id_device,
	int width,
	int depth,
	int block_width_device,
	int block_depth_device,
	int block_height_device,
	point3d * min_points_device) 
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	int i = c + r*w;
	if (c >= w || r >= h)return;
	const uchar4 background = make_uchar4(255, 255, 255, 255);//Backgound color
	float3 source = { 0,0,-zs };
	//printf("%d\n", block_data_device);
	float3 pix = scrIdxToPos(c, r, w, h, 2 * volSize.z - zs);
	//float3 pix = { c - w / 2,r - h / 2,2 * volSize.z - zs };

	source = yRotate(source, theta);
	pix = yRotate(pix, theta);
	float t0, t1;
	const Ray pixRay = { source,pix - source };
	float3 center = { volSize.x / 2.f,volSize.y / 2.f,volSize.z / 2.f }; 
	const float3 boxmin = -center;

	const float3 boxmax = {volSize.x-center.x,volSize.y-center.y,volSize.z-center.z};

	const bool hitBox = intersectBox(pixRay, boxmin, boxmax, &t0, &t1);
	uchar4 shade;
	if (hitBox == false)shade = background;
	else {
		if (t0 < 0.0f)t0 = 0.0f;
		const Ray boxRay = { paramRay(pixRay,t0),paramRay(pixRay,t1) - paramRay(pixRay,t0) };
		/*if (method == 1)
			shade = sliceShader(d_vol, volSize, boxRay, threshold, dist, source);
		else if (method == 2)
			shade = rayCastShader(d_vol, volSize, boxRay, threshold);
		else if (method == 3)
			shade = volumeRenderShader(d_vol, volSize, boxRay, threshold,NUMSTEPS);
		else 
		*/
		shade = realTimeRenderForSGMMCluster(volSize, boxRay, 
			block_data_device, 
			all_block_integrations_device,
			raw_src_device, 
			data_trans_device,
			block_id_device,width,depth,block_width_device,block_depth_device,block_height_device,min_points_device);

	}
	//printf("%d %d %d %d\n", width, depth, block_width_device, block_depth_device);
	d_out[i] = shade;
}

__global__ void volumeKernel(float *d_vol, int3 volSize, int id, float4 params) {
	const int w = volSize.x, h = volSize.y, d = volSize.z;
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	const int s = blockIdx.z*blockDim.z + threadIdx.z;
	//
	const int i = c + r*w + s*w*h;
	if (c >= w || r >= h || s >= d)return;
	d_vol[i] = func(c, r, s, id, volSize, params);
	//printf("%d", i);
}
void kernelLauncher(uchar4 * d_out, float * d_vol, int w, int h, int3 volSize, 
	int method, int zs, float theta, float threshold,float dist)
{
	dim3 blockSize(TX_2D, TY_2D);
	dim3 gridSize((w+ TX_2D - 1) / TX_2D, (h + TY_2D - 1) / TY_2D);
	//std::cout << "blockSize.x" << blockSize.x << std::endl
	//	<< "blockSize.y" << blockSize.y << std::endl
	//	<< "gridSize.x" << gridSize.x << std::endl
	//	<< "gridSize.y" << gridSize.y << std::endl
	//	<< "w:" << w << std::endl
	//	<< "h:" << h << std::endl
	//	<< "VolSize.x:" << volSize.x << std::endl
	//	<< "volSize.y:" << volSize.y << std::endl
	//	<< "volSize.z:" << volSize.z << std::endl
	//	<< "block_data_device:" << (int)block_data_device << std::endl
	//	<< "all_block_integrations_device:" << (int)all_block_integrations_device << std::endl
	//	<< "raw_src_device:" << (int)raw_src_device << std::endl
	//	<< "data_trans_device:" << (int)data_trans_device << std::endl
	//	<< "zs:" << zs << std::endl
	//	<< "theta:" << theta << std::endl
	//	<< "threshold:" << threshold << std::endl
	//	<< "dist:" << dist << std::endl;

		
	renderKernel<<<gridSize, blockSize >>>(d_out, d_vol, w, h, volSize,method,zs,theta, threshold, dist,block_data_device,all_block_integrations_device,raw_src_device,data_trans_device,block_id_device,real_width,real_depth,block_width,block_depth,block_height,min_points_device);
	//printf("%d %d %d\n", real_depth, block_width, block_depth);
	CUDA_CALL(cudaDeviceSynchronize());
}

__host__
void readResource()
{
	std::string data_source = "Combustion";
	std::string disk_address = "d:/train/";
	std::string src_raw_address = disk_address + data_source + ".raw";
	std::string sgmm_binary_address = disk_address + data_source + "_SGMM_Cluster_Result.sgmm";
	std::string integration_cluster_address = disk_address + data_source + "_integrations_sgmm_cluster";
	std::string trans_address = disk_address + data_source + "_TF.txt";
	real_width = 480; // 原始数据的真实宽
	real_depth = 720; // 原始数据的真实深
	real_height = 112; // 原始数据的真实
	int side = 16;


	std::cout << "input data address\n";
	std::cin >> disk_address;
	std::cout << "input data name\n";
	std::cin >> data_source;

#ifdef SQUARE
	std::cout << "input data width depth height side (4)\n";
	std::cin >> real_width >> real_depth >> real_height >> side;
	width_num = real_width / side;
	depth_num = real_depth / side;
	height_num = real_height / side;
	block_size = side * side *side;
	block_num = width_num * depth_num * height_num;
#else
	std::cout << "input data width depth height (3)\n";
	std::cin >> real_width >> real_depth >> real_height;
	std::cout << "input block side(3 w d h):\n";
	std::cin>> block_width >> block_depth >>block_height;

	//并不是真实的每边块的数量 而是以八叉树最小的叶节点分成的网格边长
	width_num = real_width / block_width;
	depth_num = real_depth / block_depth;
	height_num = real_height / block_height;

	std::cout << "Reading .oc file\n";
	std::ifstream read_block_num(disk_address + data_source + ".oc");
	if (read_block_num.is_open() == false){
		std::cout << "can not read .oc file\n";
		return;
	}
	read_block_num >> block_num;

	min_points = new point3d[block_num];
	for (int i = 0; i < block_num; i++){
		pos_type x, y, z, X, Y, Z;
		id_type id;
		read_block_num >> x >> y >> z >> X >> Y >> Z >> id;
		min_points[id].x = x;
		min_points[id].y = y;
		min_points[id].z = z;
	}
	//std::cout << block_num << std::endl;
	read_block_num.close();
	CUDA_CALL(cudaMalloc(&min_points_device, sizeof(point3d)*block_num));
	CUDA_CALL(cudaMemcpy(min_points_device, min_points, sizeof(point3d)*block_num, cudaMemcpyHostToDevice));
	delete[] min_points;

	int unreal_block_num = width_num*depth_num*height_num;
	block_size = block_width*block_depth*block_height;
#endif

	volumeSize.x = real_width;
	volumeSize.y = real_depth;
	volumeSize.z = real_height;


	src_raw_address = disk_address + data_source + ".raw";
	sgmm_binary_address = disk_address + data_source + "_SGMM_Cluster_Result.sgmm";
	integration_cluster_address = disk_address + data_source + "_integrations_sgmm_cluster";
	trans_address = disk_address + data_source + "_TF.txt";



	// 读取原始raw数据 test only


	raw_src_host = new unsigned char[real_width * real_depth * real_height];
	CUDA_CALL(cudaMalloc(&raw_src_device, real_width * real_depth * real_height * sizeof(unsigned char)));
	std::ifstream f_src_in(src_raw_address, std::ios::binary);
	if (f_src_in.is_open() == false) {
		std::cout << "can not open src raw file\n";
		return;
	}
	f_src_in.read((char*)(raw_src_host), real_width * real_depth * real_height);
	f_src_in.close();
	CUDA_CALL(cudaMemcpy(raw_src_device, raw_src_host, real_width * real_depth * real_height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	delete[] raw_src_host;


	// Part1: 初始化
	std::cout << "Part1: Initializing..." << std::endl;
	CUDA_CALL(cudaMalloc(&data_trans_device, 4 * 255 * sizeof(float)));
	data_trans_host = new float[4 * 255 * sizeof(float)];
	std::ifstream f_trans(trans_address);
	if (f_trans.is_open() == false) {
		std::cout << "can not open trans func file\n";
		return;
	}
	for (int i = 0; i < 4 * 255; i++) {
		f_trans >> data_trans_host[i];
	}
	f_trans.close();
	CUDA_CALL(cudaMemcpy(data_trans_device, data_trans_host, 4 * 255 * sizeof(float), cudaMemcpyHostToDevice));
	delete[] data_trans_host;

	// Part2: 读取sgmm
	std::cout << std::endl << "Part2: Reading SGMMs..." << std::endl;
	CUDA_CALL(cudaMalloc(&block_data_device, block_num * sizeof(Block)));
	block_data_host = new Block[block_num];
	std::ifstream f_sgmm(sgmm_binary_address, std::ios::binary);
	if (f_sgmm.is_open() == false) {
		std::cout << "can not open sgmm file\n";
		return;
	}
	for (int block_index = 0; block_index < block_num; block_index++) {
		f_sgmm.read((char*)&(block_data_host[block_index].cluster_num), sizeof(unsigned char));
		for (int cluster_index = 0; cluster_index < block_data_host[block_index].cluster_num; cluster_index++) {
			f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].probability_), sizeof(float));
			f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gauss_count_), sizeof(unsigned char));
			f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].sample_value_), sizeof(float));
			for (int gauss_index = 0; gauss_index < block_data_host[block_index].clusters_[cluster_index].gauss_count_; gauss_index++) {
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].weight_), sizeof(float));
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].determinant_), sizeof(float));
				for (int i = 0; i < 3; i++) {
					f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].mean_[i]), sizeof(float));
				}
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[0]), sizeof(float));
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[1]), sizeof(float));
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[2]), sizeof(float));
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[4]), sizeof(float));
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[5]), sizeof(float));
				f_sgmm.read((char*)&(block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[8]), sizeof(float));
				block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[3] = block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[1];
				block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[6] = block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[2];
				block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[7] = block_data_host[block_index].clusters_[cluster_index].gausses_[gauss_index].precisions_[5];
			}
		}
	}
	f_sgmm.close(); 
	CUDA_CALL(cudaMemcpy(block_data_device, block_data_host, block_num * sizeof(Block), cudaMemcpyHostToDevice));
	//printf("%d ---\n", block_data_device);
	delete[] block_data_host;

	// Part3: 读取积分
	std::cout << std::endl << "Part3: Reading integrations..." << std::endl;
	CUDA_CALL(cudaMalloc(&all_block_integrations_device, block_num * sizeof(Integrations)));
	all_block_integrations_host = new Integrations[block_num];
	std::ifstream f_in(integration_cluster_address, std::ios::binary);
	if (f_in.is_open() == false) {
		std::cout << "can not open integrations file\n";
		return;
	}
	for (int i = 0; i < block_num; i++) {
		for (int j = 0; j < MAX_CLUSTER_NUM; j++) {
			f_in.read((char*)&(all_block_integrations_host[i].integration_value[j]), sizeof(float));
		}
	}
	CUDA_CALL(cudaMemcpy(all_block_integrations_device, all_block_integrations_host, block_num * sizeof(Integrations), cudaMemcpyHostToDevice));
	delete[] all_block_integrations_host;

	//Part4:读取索引
	std::cout << std::endl << "Part4:Reading exoc file" << std::endl;
	block_id_host = new id_type[unreal_block_num];
	std::ifstream ocex_in_file(disk_address + data_source + ".ocex");
	if (ocex_in_file.is_open() == false){
		std::cout << "can not open .ocex file\n";
		return;
	}
	for (int i = 0; i < unreal_block_num; i++){
		ocex_in_file >> block_id_host[i];
	}

	//这里的unreal_block_num 可能与文件里的元素数量不匹配

	ocex_in_file.close();
	CUDA_CALL(cudaMalloc(&block_id_device, sizeof(id_type)*unreal_block_num));
	CUDA_CALL(cudaMemcpy(block_id_device, block_id_host, sizeof(id_type)*unreal_block_num, cudaMemcpyHostToDevice));
	delete[] block_id_host;


	//Part4:读取每块起始坐标
	
}




void volumeKernelLancher(float * d_vol, int3 volSize, int id, float4 params)
{
	//dim3 blockSize(TX, TY, TZ);
	//dim3 gridSize((volSize.x + TX - 1) / TX, (volSize.y + TY - 1) / TY, (volSize.z + TZ - 1) / TZ);
	//volumeKernel<<<gridSize, blockSize>>> (d_vol, volSize, id, params);
	calcExp(.01f, -10.f, 40.f);
	readResource();
}