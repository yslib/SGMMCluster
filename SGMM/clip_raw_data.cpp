#include "funcs.h"
int clip_raw_data(int argc, char ** argv) {
	std::cout << "------------CLIP RAW DATA MODULE----------\n";
	std::cout << "input data address:\n";
	std::string path;
	std::cin >> path;
	std::cout << "input data name:\n";
	std::string data_name;
	std::cin >> data_name;

	int width, depth, height;
	std::cout << "input width depth height(3):\n";
	std::cin >> width >> depth >> height;

	std::size_t raw_size = get_file_size(path + data_name + ".raw");
	std::size_t new_size = width*depth*height;
	if (new_size > raw_size) {
		std::cout << "size error\n";
		return 0;
	}
	voxel_type * data = new voxel_type[raw_size];
	if (data == nullptr) {
		std::cout << "allocating memory failed\n";
		return 0;
	}
	
	voxel_type * cliped_data = new voxel_type[new_size];
	if (cliped_data == nullptr) {
		delete[] data;
		std::cout << "allocating memory for cliped data failed\n";
		return 0;
	}

	for (int z = 0; z < height; z++) {
		for (int y = 0; y < depth; y++) {
			for (int x = 0; x < width; x++) {
				int index = x + y*width + z*width*depth;
				cliped_data[index] = data[index];
			}
		}
	}
	std::ofstream out_file(path + data_name + "_resize.raw",std::ios::binary);
	if (out_file.is_open() == false) {
		std::cout << "Creating raw file failed\n";
		delete[] data;
		delete[] cliped_data;
		return 0;
	}
	out_file.write((const char *)cliped_data, new_size);
	out_file.close();
	delete[] data;
	delete[] cliped_data;
	return 0;
}