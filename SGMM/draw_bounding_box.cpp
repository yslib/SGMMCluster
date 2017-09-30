#include <string>
#include <fstream>
#include <iostream>
#include "funcs.h"

//Draw a bounding box for a given min-max coordinate.
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
int create_AABB_file(int argc, char ** argv) {
	std::string disk_address;
	std::string aabb_file;
	int width;
	int depth;
	int height;
	int side;

	std::cout << "----------CREATE AABB FILE MODULE------------\n";
	std::cout << "input data address\n";
	std::cin >> disk_address;
	std::cout << "input aabb file\n";
	std::cin >> aabb_file;
	std::cout << "input width depth height (3)\n";
	std::cin >> width >> depth >> height >> side;

	if (create_AABB_file(disk_address, aabb_file, create_regular_boundingbox(width, depth, height, side, side, side)) == false) {
		std::cout << "Creating aabb file for sgmm and block gmm failed\n";
		exit(1);
	}
	std::cin.get();
	return 0;
}

//Function Entry
int draw_bounding_box(int argc, char ** argv) {

	std::string disk_address;
	std::string data_source;
	std::string data_target;
	std::string aabb_file;
	int width, depth, height;

	//parameter input
	std::cout << "----------DRAW BOUNDING BOX MODULE------------\n";
	std::cout << "input data address\n";
	std::cin >> disk_address;
	std::cout << "input data name\n";
	std::cin >> data_source;
	std::cout << "input target name\n";
	std::cin >> data_target;
	std::cout << "input aabb file\n";
	std::cin >> aabb_file;
	std::cout << "input width depth height(3)\n";
	std::cin >> width >> depth >> height;

	//reading volume data
	voxel_type * volume_data = new voxel_type[width*depth*height];
	if (read_raw_file(disk_address + data_source + ".raw", volume_data, width, depth, height) == false) {
		std::cout << "can not read .raw file\n";
		exit(1);
	}

	//Draw aabbs
	std::vector<AABB> aabbs = read_AABB_from_file(disk_address, aabb_file);
	for (const AABB & aabb : aabbs) {
		draw_bounding_box(volume_data, aabb.min_point, aabb.max_point, width, depth, height);
	}

	//Writing new volume data
	std::ofstream new_raw_file(disk_address + data_target+"_AABB" + ".raw");
	if (new_raw_file.is_open() == false) {
		std::cout << "can not create .raw file\n";
		exit(1);
	}
	new_raw_file.write((const char *)volume_data, width*depth*height);
	std::cout << "Drawing bounding box finished\n";
	delete[] volume_data;

	//Creating .vifo file for new volume data
	std::cout << "Creating .vifo file for _AABB.raw file\n";
	create_vifo_file(disk_address, data_target + "_AABB", width, depth, height);
	std::cin.get();
	return 0;
}