#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <string>
#include <iostream>
#include "funcs.h"

using namespace std;

int float2byte(int argc,char ** argv)
{
	//-----------------FLOAT2BYTE MODEUL------------------
	int xiSize = 480, yiSize = 720, ziSize = 120;
	std::cout << "input data address\n";
	std::string data_address;
	std::cin >> data_address;
	std::cout << "input file name (full name)\n";
	std::string file_name;
	std::cin >> file_name;
	std::cout << "input x y z(3):\n";
	std::cin >> xiSize >> yiSize >> ziSize;
	FILE* fp = fopen((data_address+file_name).c_str(), "rb");
	if (fp == nullptr) {
		std::cout << "can not open file\n";
		return 0;
	}
	unsigned char* pOriginalData = new unsigned char[xiSize * yiSize * ziSize * 4];
	if (pOriginalData == NULL) {
		fclose(fp);
		delete[]pOriginalData;
		pOriginalData = NULL;
		cout << "Out of memory " << xiSize * yiSize * ziSize / 1024.0 / 1024.0 << " M" << endl;
		return false;
	}
	fread(pOriginalData, sizeof(unsigned char), xiSize * yiSize * ziSize * 4, fp);
	fclose(fp);

	//for (int i = 0; i < xiSize * yiSize * ziSize; i++)
	//{
	//	swap(pOriginalData[4 * i], pOriginalData[4 * i + 3]);
	//	swap(pOriginalData[4 * i + 1], pOriginalData[4 * i + 2]);
	//}

	float* convertedData = (float *)pOriginalData;

	// ¼ÆËã×î´ó×îÐ¡
	float max_value = convertedData[0];
	float min_value = convertedData[0];
	int data_size = xiSize * yiSize * ziSize;
	for (int i = 1; i < data_size; i++) {
		if (convertedData[i] < -2.7e-19) {
			continue;
		}

		if (convertedData[i] > max_value) {
			max_value = convertedData[i];
		}
		if (convertedData[i] < min_value) {
			min_value = convertedData[i];
		}

	}

	std::cout << "max value = " << max_value << ", min value = " << min_value << std::endl;

	float maxScalar = 3225.42578, minScalar = -5471.85791;
	unsigned char* normalizedData = new unsigned char[xiSize * yiSize * ziSize];
	for (int i = 0; i < xiSize * yiSize * ziSize; i++)
	{
		if (convertedData[i] > maxScalar || convertedData[i] < minScalar)
			normalizedData[i] = 0;
		else
			normalizedData[i] = (unsigned char)((convertedData[i] - minScalar) / (maxScalar - minScalar) * 255.0);
	}

	FILE* fwp = fopen((data_address+file_name+".raw").c_str(), "wb");
	fwrite(normalizedData, sizeof(unsigned char), xiSize * yiSize * ziSize, fwp);
	fclose(fwp);
	create_vifo_file(data_address, file_name + ".vifo", xiSize, yiSize, ziSize);
	std::cin.get();
	return 0;
}