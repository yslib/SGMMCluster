#pragma once
struct uchar4;
struct int2;
struct int3;
struct float4;
void kernelLauncher(uchar4 * d_out, float *d_vol, int w, int h, int3 volSize,
	int method, int zs, float theta, float threshold,float dist);
void volumeKernelLancher(float *d_vol,int3 volSize,int id,float4 params);


