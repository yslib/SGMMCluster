#include "funcs.h"

//
//Convert int to string if int is valid
//
std::string int_to_string(int i) {
	std::stringstream s_temp;
	std::string s;
	s_temp << i;
	s_temp >> s;
	return s;
}
//
//Read raw file by byte
//
bool read_raw_file(const std::string & file_name, voxel_type * vol, size_t width, size_t depth, size_t height) {
	std::ifstream in_file(file_name, std::ios::binary);
	if (in_file.is_open() == false) {
		std::cout << "can not open raw file\n";
		return false;
	}
	if (!in_file.read((char *)vol, width*depth*height * sizeof(voxel_type))) {
		std::cout << "Reading .raw file failed." << std::endl;
		throw std::out_of_range("Reading .raw file error\n");
	}
	return true;
}

//
//Create a .vifo file 
//
bool create_vifo_file(const std::string & address, const std::string & file_name, int width, int depth, int height)
{
	std::ofstream out_vifo_file(address + file_name + ".vifo");
	if (out_vifo_file.is_open() == false) {
		std::cout << "Creating .vifo file failed\n";
		return false;
	}
	out_vifo_file << width << " " << depth << " " << height << std::endl;	//size of the volume data
	out_vifo_file << "1 1 1" << std::endl;
	out_vifo_file << file_name << ".vifo" << std::endl;		//path of the .raw file(assuming both are the same directory)
	out_vifo_file.close();
	return true;
}

std::vector<AABB> read_AABB_from_file(const std::string path, const std::string & file_name) {
	std::vector<AABB> aabbs;
	std::ifstream bounding_box_file(path + file_name);
	if (bounding_box_file.is_open() == false) {
		std::cout << "can not open bounding box file\n";
		exit(1);
	}
	int block_num;
	bounding_box_file >> block_num;
	AABB aabb;
	for (int i = 0; i < block_num; i++) {
		//bounding_box_file >> aabb.max_point.x >> aabb.min_point.y >> aabb.min_point.z >> aabb.max_point.x >> aabb.max_point.y >> aabb.max_point.z;
		bounding_box_file >> aabb.min_point >> aabb.max_point;
		while (bounding_box_file.get() != '\n')continue;		//empty the rest of current line
		aabbs.push_back(aabb);
	}
	return aabbs;
}

std::vector<AABB> create_regular_boundingbox(
	int width,
	int depth,
	int height,
	int block_width,
	int block_depth,
	int block_height)
{
	assert(width%block_width == 0);
	assert(depth%block_depth == 0);
	assert(height%block_height == 0);

	int block_width_num = width / block_width;
	int block_depth_num = depth / block_depth;
	int block_height_num = height / block_height;
	std::vector<AABB> bounding_boxes;
	for (int z = 0; z < block_height_num; z++) {
		for (int y = 0; y < block_depth_num; y++) {
			for (int x = 0; x < block_width_num; x++) {
				auto minx = x*block_width;
				auto miny = y*block_depth;
				auto minz = z*block_height;
				auto maxx = minx + block_width;
				auto maxy = miny + block_depth;
				auto maxz = minz + block_height;
				bounding_boxes.emplace_back(point3d(minx, miny, minz), point3d(maxx, maxy, maxz));
			}
		}
	}
	return bounding_boxes;
}

//
bool create_AABB_file(const std::string & path, const std::string & file_name, const std::vector<AABB>& aabbs)
{
	std::ofstream aabb_file(path + file_name);
	if (aabb_file.is_open() == false) {
		std::cout << "Creating aabb file failed\n";
		return false;
	}
	aabb_file << aabbs.size() << std::endl;
	for (const AABB & aabb : aabbs) {
		aabb_file << aabb.min_point << " " << aabb.max_point << std::endl;
	}
	return true;
}
