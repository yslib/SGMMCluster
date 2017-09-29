#include <string>
#include <fstream>
#include <iostream>
#include "structuresfunctions.h"

void draw_bounding_box(voxel_type * voxel, const point3d & min_point,const point3d & max_point,int width,int depth,int height) {
	auto xmin = min_point.x;
	auto ymin = min_point.y;
	auto zmin = min_point.z;
	auto xmax = max_point.x;
	auto ymax = max_point.y;
	auto zmax = max_point.z;

	pos_type offset[] = { 
	xmin,xmax - 1,ymin,zmin,
	xmin,xmax - 1,ymax-1,zmin,
	xmin,xmax-1,ymin,zmax-1,
	xmin,xmax-1,ymax-1,zmax-1,

	ymin,ymax - 1,xmin,zmin,
	ymin,ymax - 1,xmax - 1,zmin,
	ymin,ymax - 1,xmin,zmax - 1,
	ymin,ymax - 1,xmax - 1,zmax - 1,

	zmin,zmax - 1,xmin,ymin,
	zmin,zmax - 1,xmax - 1,ymin,
	zmin,zmax - 1,xmin,ymax - 1,
	zmin,zmax - 1,xmax - 1,ymax - 1
	};

	for (int i = 0; i < 4; i++) {
		for (int x = offset[4*i]; x < offset[4*i+1]; x++) {
			int index = x + offset[4*i+2]*width + offset[4*i+3]*width*depth;
			voxel[index] = 255;
		}
	}
	for (int i = 4; i < 8; i++) {
		for (int y = offset[4 * i]; y < offset[4 * i + 1]; y++) {
			int index = offset[4 * i + 2] + y*width+offset[4 * i + 3] * width*depth;
			voxel[index] = 255;
		}
	}
	for (int i = 8; i < 12; i++) {
		for (int z = offset[4 * i]; z < offset[4 * i + 1]; z++) {
			int index = offset[4 * i + 2] + offset[4 * i + 3] * width + z*width*depth;
			voxel[index] = 255;
		}
	}
}

int draw_bounding_box(int argc, char ** argv) {

	std::string data_address;
	std::string data_source;
	point3d min_point;
	point3d max_point;
	int width, depth, height;

	int data_type;	/*
					* 1 for block gmm
					* 2 for sgmm
					* 3 for sgmm cluster
					*/
	std::cout << "----------DRAW BOUNDING BOX MODULE------------\n";
	std::cout << "input data address\n";
	std::cin >> data_address;
	std::cout << "input data name\n";
	std::cin >> data_source;
	std::cout << "input width depth height (3)\n";
	std::cin >> width >> depth >> height;
	std::cout << "input data type\n";
	std::cin >> data_type;
	if (data_type <= 0 || data_type > 3) {
		exit(1);
	}
	if (data_type != 3) {

	}

	//open bounding box file
	std::ifstream bounding_box_file(data_address + data_source + ".reoc");
	if (bounding_box_file.is_open() == false) {
		std::cout << "can not open .reoc file\n";
		exit(1);
	}

	//reading volume data
	voxel_type * volume_data = new voxel_type[width*depth*height];
	if (read_raw_file(data_address + data_source + ".raw", volume_data, width, depth, height) == false) {
		std::cout << "can not read .raw file\n";
		exit(1);
	}

	int block_num;
	bounding_box_file >> block_num;
	for (int i = 0; i < block_num; i++) {
		bounding_box_file >> min_point.x >> min_point.y >> min_point.z >> max_point.x >> max_point.y >> max_point.z;
		int t;
		bounding_box_file >> t;
		draw_bounding_box(volume_data, min_point, max_point, width, depth, height);
	}

	std::ofstream new_raw_file(data_address + data_source+"_AABB" + ".raw");
	if (new_raw_file.is_open() == false) {
		std::cout << "can not create .raw file\n";
		exit(1);
	}
	new_raw_file.write((const char*)volume_data, width*depth*height);
	std::cout << "Drawing bounding box finished\n";
	//delete[] volume_data;
	std::cout << "Creating .vifo file for _AABB.raw file\n";
	create_vifo_file(data_address, data_source + "_AABB", width, depth, height);
	std::cin.get();
	return 0;
}