#include "interaction.h"


int3 volumeSize = { NX,NY,NZ };
float4 params = { NX / 4.f,NY / 6.f,NZ / 16.f ,1.f };

void keyborad(unsigned char key, int x, int y) {
	if (key == 'a')dragMode = !dragMode;
	if (key == '+')zs -= DELTA;
	if (key == '-')zs += DELTA;
	if (key == 'd') { dist += 1; }
	if (key == 'D') { dist -= 1; }
	if (key == 'z') {
		zs = NZ;
		theta = 0.f;
		dist = 0.f;
	}
	if (key == 'v')method = 0;
	if (key == 'f')method = 1;
	if (key == 'r')method = 2;

	if (key == 27)std::exit(0);
	printf("key pressed\n");
	glutPostRedisplay();
}
void mouseMove(int x, int y) {
	if (dragMode == true)return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}
void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_LEFT)theta -= 0.1f;
	if (key == GLUT_KEY_RIGHT)theta += 0.1f;
	if (key == GLUT_KEY_UP)threshold += 0.1f;
	if (key == GLUT_KEY_DOWN)threshold -= 0.1f;
	printf("Special key pressed\n");
	glutPostRedisplay();
}
void printInstrctions() {
	std::printf("......\n");
}

void createMenu()
{
	glutCreateMenu(mymenu);
	glutAddMenuEntry("1", 0);
	glutAddMenuEntry("2", 1);
	glutAddMenuEntry("3", 2);
	glutAddMenuEntry("4", 3);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void mymenu(int val)
{
	switch (val)
	{
	case 0:return;
	case 1:id = 0; break;
	case 2:id = 1; break;
	case 3:id = 2; break;
	default:
		break;
	}
	volumeKernelLancher(d_vol, volumeSize, id, params);
	glutPostRedisplay();
}
