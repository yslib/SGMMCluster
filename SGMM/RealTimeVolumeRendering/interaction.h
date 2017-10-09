#pragma once
#include "cuda_runtime.h"
#include "realtimerendering.h"
#include <GL\glew.h>
#include <GL\glut.h>
#include <cstdlib>
#include <cstdio>

//Window Parameter
const int W = 600;			//Width of the picture
const int H = 600;			//Height of the picture
const int DELTA = 5;
const int NX = 128, NY = 128, NZ = 256;


extern int2 loc;
extern bool dragMode;
extern int id;
extern int method;
extern float *d_vol;
extern float zs, dist, theta, threshold;

extern int3 volumeSize;
extern float4 params;

void keyborad(unsigned char key, int x, int y);
void mouseMove(int x, int y);
void handleSpecialKeypress(int key, int x, int y);
void printInstrctions();
void createMenu();
void mymenu(int val);