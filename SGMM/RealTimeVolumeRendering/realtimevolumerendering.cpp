#include <cstdlib>
#include <iostream>
#include "realtimerendering.h"
#include "interaction.h"
#include "GL\glew.h"
#include <ctime>
#include <string>
#include <sstream>
//#include <GL\freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource * cuda_pbo_resource;
int2 loc = { W / 2,H / 2 };
bool dragMode = false;
int id = 1;
int method = 1;
float *d_vol;
float zs = NZ;
float dist = 0.0f, theta = 0.f, threshold = 0.0f;

int fps = 0;
clock_t prev_clock, pres_clock;

static std::string Int2String(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}
void updateFPS() {
	fps++;
	pres_clock = clock();
	if ((pres_clock - prev_clock) / CLOCKS_PER_SEC >= 1) {
		glutSetWindowTitle(Int2String(fps).c_str());
		fps = 0;
		prev_clock = pres_clock;
	}
}

void render()
{
	uchar4 *d_out = 0;

	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);

	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);

	kernelLauncher(d_out,d_vol, W, H,volumeSize,method,zs,theta,threshold,dist);

	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

	//printf("Rendered\n");

}

void drawTexture() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,GL_UNSIGNED_BYTE,NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}
void display() {
	render();
	drawTexture();
	glutSwapBuffers();
	updateFPS();
	//glFlush();

}
void initGLUT(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(W, H);
	glutCreateWindow("asdfa");
	glewInit();
}

void initPixelBuffer() {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H * sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}
void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}
int RealTimeVolumeRender(int argc, char ** argv) {
	//cudaMalloc(&d_vol, NX*NY*NZ * sizeof(float));
	prev_clock = clock();
	volumeKernelLancher(d_vol, volumeSize, id, params);
	printInstrctions();
	initGLUT(argc, argv);
	initPixelBuffer();
	createMenu();
	gluOrtho2D(0, W, 0, H);
	glutKeyboardFunc(keyborad);
	glutSpecialFunc(handleSpecialKeypress);
	glutPassiveMotionFunc(mouseMove);
	glutDisplayFunc(display);
	glutMainLoop();
	atexit(exitfunc);
	return 0;
}