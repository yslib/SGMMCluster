#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <exception>
#include "commands.h"
#include "funcs.h"
#include <ctime>

#define LEAF_CHECK
#define LEAF_ID_CONTINUOUS_CHECK

#define INPUT

static std::size_t width;
static std::size_t depth;
static std::size_t height;
static std::size_t data_width;
static std::size_t data_depth;
static std::size_t data_height;
static std::size_t min_side;

static voxel_type * volume_data;

const int max_cluster_num = 12;
const int min_cluster_num = 6;

//基础数据结构
//3D point



//OCtree Node

struct octree_node {

	point3d min_point;
	point3d max_point;
	id_type id;
	octree_node* children[8];
	octree_node* parent;
	bool valid;
	point3d max_data_point;
	point3d min_data_point;
	double ent;


	octree_node(pos_type x, pos_type y, pos_type z, pos_type X, pos_type Y, pos_type Z, bool v = true, id_type i = -1) :
		min_point{ x, y, z }, max_point{ X, Y, Z }, id{ i }, children{}, parent{}, valid{ v }, min_data_point{ x, y, z }, max_data_point{ X, Y, Z }, ent{ 0.0 }
	{}
	octree_node(pos_type x, pos_type y, pos_type z, pos_type X, pos_type Y, pos_type Z, pos_type d_x, pos_type d_y, pos_type d_z, pos_type d_X, pos_type d_Y, pos_type d_Z, bool v = true, id_type i = -1) :
		min_point{ x, y, z }, max_point{ X, Y, Z }, min_data_point{ d_x, d_y, d_z }, max_data_point{ d_X, d_Y, d_Z }, id{ i }, children{}, valid{ v }, parent{}, ent{ 0.0 }
	{}
	octree_node(const point3d & start, const point3d & end, id_type i = -1) :
		octree_node(start.x, start.y, start.z, end.x, end.y, end.z, i)
	{}
	octree_node(const point3d & start, const point3d & end, const point3d & data_start, const point3d & data_end, bool v = true, id_type i = -1) :
		octree_node(start.x, start.y, start.z, end.x, end.y, end.z, data_start.x, data_start.y, data_start.z, data_end.x, data_end.y, data_end.z, v, i)
	{}

};

//

id_type ids = 0;
inline id_type new_id() { return ids++; }
inline void id_clear() { ids = 0; }


static std::vector<octree_node*> octree_nodes;


//some functors for subdividing criterion and leaf sorting

//Simple criterion
struct min_max {
private:
	int threshold;
public:
	min_max(int t) :threshold{ t } {}
	min_max() = delete;
	bool operator()(voxel_type * vol, const octree_node * root) {
		auto xmin = root->min_point.x;
		auto ymin = root->min_point.y;
		auto zmin = root->min_point.z;
		auto xmax = root->max_point.x;
		auto ymax = root->max_point.y;
		auto zmax = root->max_point.z;
		auto min_val = vol[xmin + ymin*depth + zmin*width*depth];
		auto max_val = vol[xmin + ymin*depth + zmin*width*depth];
		for (auto z = zmin; z < zmax; z++) {
			for (auto y = ymin; y < ymax; y++) {
				for (auto x = xmin; x < xmax; x++) {
					int index = x + y*width + z*width*depth;
					if (min_val > vol[index])min_val = vol[index];
					if (max_val < vol[index])max_val = vol[index];
				}
			}
		}
		return max_val - min_val >= threshold;
	}
};

//information entropy criterion
const int EVENT_NUM = 128;
struct entropy {
private:
	double threshold;
	std::size_t _width;
	std::size_t _depth;
	std::size_t _height;
	std::size_t _size;
public:
	entropy(voxel_type * vol, std::size_t width, std::size_t depth, std::size_t height, double t) :threshold{ t },
		_width{ width }, _depth{ depth }, _height{ height }, _size{ width*depth*height } {
	}
	entropy() = delete;
	bool operator()(voxel_type * vol, const octree_node * root) {
		auto xmin = root->min_data_point.x;
		auto ymin = root->min_data_point.y;
		auto zmin = root->min_data_point.z;
		auto xmax = root->max_data_point.x;
		auto ymax = root->max_data_point.y;
		auto zmax = root->max_data_point.z;
		//std::cout << zmax << std::endl;
		double ent = 0.0;
		int count[EVENT_NUM];
		std::memset(count, 0, sizeof(int) * EVENT_NUM);
		int total = (xmax - xmin)*(ymax - ymin)*(zmax - zmin);
		for (auto z = zmin; z < zmax; z++) {
			for (auto y = ymin; y < ymax; y++) {
				for (auto x = xmin; x < xmax; x++) {
					int index = x + y*data_width + z*data_width*data_depth;
					count[(unsigned char)vol[index] / 2]++;
				}
			}
		}
		//for (auto z = zmin; z < zmax; z++) {
		//	for (auto y = ymin; y < ymax; y++) {
		//		for (auto x = xmin; x < xmax; x++) {
		//			int index = x + y*data_width + z*data_width*data_depth;
		//			int scalar = (unsigned char)vol[index];
		//			double prob = static_cast<double>(count[scalar]) / total;
		//			ent += prob * std::log(prob) / std::log(2);
		//		}
		//	}
		//}
		for (int i = 0; i < EVENT_NUM; i++) {
			if (count[i] != 0) {
				double prob = static_cast<double>(count[i]) / total;
				double logprob = std::log(prob) / std::log(2);
				ent += prob *logprob;
			}

		}
		const_cast<octree_node *>(root)->ent = -ent;
		//std::cout << -ent << std::endl;
		return (-ent >= threshold);
	}
};
bool leaf_cmp(const octree_node *a, const octree_node *b) {
	if (a->min_point.z != b->min_point.z)return a->min_point.z < b->min_point.z;
	else if (a->min_point.y != b->min_point.y)return a->min_point.y < b->min_point.y;
	else if (a->min_point.x != b->min_point.x)return a->min_point.x < b->min_point.x;
	else return false;
}


bool is_leaf(const octree_node * root) {
	bool leaf = true;
	for (int i = 0; i < 8; i++) {
		if (root->children[i] != nullptr) {
			leaf = false;
			break;
		}
	}
	return leaf;
}
bool is_power_of_2(int x) { return (x&(-x)) == x; }
int next_pow_of_2(int x) {
	if (is_power_of_2(x) == true)return x;
	int next = 1;
	while (x != 0) {
		next <<= 1;
		x >>= 1;
	}
	return next;
}

//Create a OCTree

template<typename functor>
void create_octree(voxel_type * vol, octree_node * &root, size_t mside, functor fcr) {
	if (root == nullptr) {            //root node
		root = new octree_node(0, 0, 0, width, depth, height);
		octree_nodes.push_back(root);
	}

	//boundary condition
	if (fcr(vol, root) == false || root->max_point.x - root->min_point.x <= mside
		|| root->max_point.y - root->min_point.y <= mside || root->max_point.z - root->min_point.z <= mside) {
		// 1.criterion 2.minimum block
		return;
	}

	octree_node * new_node;
	pos_type dx = (root->max_point.x - root->min_point.x);
	pos_type dy = (root->max_point.y - root->min_point.y);
	pos_type dz = (root->max_point.z - root->min_point.z);
	pos_type dx1 = dx / 2, dx2;
	pos_type dy1 = dy / 2, dy2;
	pos_type dz1 = dz / 2, dz2;
	if (dx % 2 == 1)dx2 = dx1 + 1; else dx2 = dx1;
	if (dy % 2 == 1)dy2 = dy1 + 1; else dy2 = dy1;
	if (dz % 2 == 1)dz2 = dz1 + 1; else dz2 = dz1;
	point3d offset[8][2] = { { point3d(0, 0, 0), point3d(dx1, dy1, dz1) },
	{ point3d(dx1, 0, 0), point3d(dx1 + dx2, dy1, dz1) },
	{ point3d(0, dy1, 0), point3d(dx1, dy1 + dy2, dz1) },
	{ point3d(dx1, dy1, 0), point3d(dx1 + dx2, dy1 + dy2, dz1) },
	{ point3d(0, 0, dz1), point3d(dx1, dy1, dz1 + dz2) },
	{ point3d(dx1, 0, dz1), point3d(dx1 + dx2, dy1, dz1 + dz2) },
	{ point3d(0, dy1, dz1), point3d(dx1, dy1 + dy2, dz1 + dz2) },
	{ point3d(dx1, dy1, dz1), point3d(dx1 + dx2, dy1 + dy2, dz1 + dz2) } };

	//children node distribute first along x axis,and then y axis ...
	for (int z = 0; z < 2; z++) {
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 2; x++) {
				int i = x + y * 2 + z * 4;
				new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1]);
				octree_nodes.push_back(new_node);
				new_node->parent = root;
				octree_node *& ref = root->children[x + y * 2 + 4 * z];
				ref = new_node;
				create_octree(vol, ref, mside, fcr);
			}
		}
	}
}

template<typename functor>
void create_regular_octree(voxel_type * vol, octree_node * &root, size_t mside, functor fcr) {
	if (root == nullptr) {            //root node
		assert(is_power_of_2(width) && is_power_of_2(depth) && is_power_of_2(height));
		root = new octree_node(0, 0, 0, width, depth, height, 0, 0, 0, data_width, data_depth, data_height, true);
		octree_nodes.push_back(root);
	}

	//check whether the block is empty
	if (root->max_data_point.x - root->min_data_point.x == 0 || root->max_data_point.y - root->min_data_point.y == 0 || root->max_data_point.z - root->min_data_point.z == 0)
		root->valid = false;

	//boundary condition
	if (fcr(vol, root) == false || root->valid == false || root->max_point.x - root->min_point.x <= mside
		|| root->max_point.y - root->min_point.y <= mside || root->max_point.z - root->min_point.z <= mside) {
		// 1.criterion 2.minimum block
		return;
	}

	octree_node * new_node;
	pos_type dx = (root->max_point.x - root->min_point.x);
	pos_type dy = (root->max_point.y - root->min_point.y);
	pos_type dz = (root->max_point.z - root->min_point.z);
	assert(dx % 2 == 0);
	assert(dy % 2 == 0);
	assert(dz % 2 == 0);
	pos_type dx1 = dx / 2;
	pos_type dy1 = dy / 2;
	pos_type dz1 = dz / 2;
	pos_type dx2 = dx1;
	pos_type dy2 = dy1;
	pos_type dz2 = dz1;

	//size of octree
	point3d offset[8][2] = { { point3d(0, 0, 0), point3d(dx1, dy1, dz1) },
	{ point3d(dx1, 0, 0), point3d(dx1 + dx2, dy1, dz1) },
	{ point3d(0, dy1, 0), point3d(dx1, dy1 + dy2, dz1) },
	{ point3d(dx1, dy1, 0), point3d(dx1 + dx2, dy1 + dy2, dz1) },
	{ point3d(0, 0, dz1), point3d(dx1, dy1, dz1 + dz2) },
	{ point3d(dx1, 0, dz1), point3d(dx1 + dx2, dy1, dz1 + dz2) },
	{ point3d(0, dy1, dz1), point3d(dx1, dy1 + dy2, dz1 + dz2) },
	{ point3d(dx1, dy1, dz1), point3d(dx1 + dx2, dy1 + dy2, dz1 + dz2) } };

	//size of volume data
	bool subdivide[8] = { true, true, true, true, true, true, true, true };
	pos_type d_dx = (root->max_data_point.x - root->min_data_point.x);
	pos_type d_dy = (root->max_data_point.y - root->min_data_point.y);
	pos_type d_dz = (root->max_data_point.z - root->min_data_point.z);

	pos_type d_dx1 = std::min(d_dx, dx1);
	pos_type d_dy1 = std::min(d_dy, dy1);
	pos_type d_dz1 = std::min(d_dz, dz1);
	pos_type d_dx2 = std::max(d_dx - d_dx1, 0);
	pos_type d_dy2 = std::max(d_dy - d_dy1, 0);
	pos_type d_dz2 = std::max(d_dz - d_dz1, 0);
	if (d_dx2 == 0) {
		subdivide[1] = false;
		subdivide[3] = false;
		subdivide[5] = false;
		subdivide[7] = false;
	}
	if (d_dy2 == 0) {
		subdivide[2] = false;
		subdivide[3] = false;
		subdivide[6] = false;
		subdivide[7] = false;
	}
	if (d_dz2 == 0) {
		subdivide[4] = false;
		subdivide[5] = false;
		subdivide[6] = false;
		subdivide[7] = false;
	}

	point3d data_offset[8][2] = { { point3d(0, 0, 0), point3d(d_dx1, d_dy1, d_dz1) },
	{ point3d(d_dx1, 0, 0), point3d(d_dx1 + d_dx2, d_dy1, d_dz1) },
	{ point3d(0, d_dy1, 0), point3d(d_dx1, d_dy1 + d_dy2, d_dz1) },
	{ point3d(d_dx1, d_dy1, 0), point3d(d_dx1 + d_dx2, d_dy1 + d_dy2, d_dz1) },
	{ point3d(0, 0, d_dz1), point3d(d_dx1, d_dy1, d_dz1 + d_dz2) },
	{ point3d(d_dx1, 0, d_dz1), point3d(d_dx1 + d_dx2, d_dy1, d_dz1 + d_dz2) },
	{ point3d(0, d_dy1, d_dz1), point3d(d_dx1, d_dy1 + d_dy2, d_dz1 + d_dz2) },
	{ point3d(d_dx1, d_dy1, d_dz1), point3d(d_dx1 + d_dx2, d_dy1 + d_dy2, d_dz1 + d_dz2) } };




	//children node distribute first along x axis,and then y axis ...
	for (int z = 0; z < 2; z++) {
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 2; x++) {
				int i = x + y * 2 + z * 4;
				if (subdivide[i] == true) {
					new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1], root->min_data_point + data_offset[i][0], root->min_data_point + data_offset[i][1], subdivide[i]);

				}
				else {
					new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1], root->min_point + offset[i][0], root->min_point + offset[i][0], subdivide[i]);
				}
				octree_nodes.push_back(new_node);
				new_node->parent = root;
				octree_node *& ref = root->children[x + y * 2 + 4 * z];
				ref = new_node;
				create_regular_octree(vol, ref, mside, fcr);

			}
		}
	}
}

//read volume data from file
bool read_data(const std::string & file_name, voxel_type * vol, size_t width, size_t depth, size_t height) {
	std::ifstream in_file(file_name, std::ios::binary);
	if (in_file.is_open() == false) {
		std::cout << "can not open raw file\n";
		return false;
	}
	if (!in_file.read((char *)vol, width*depth*height * sizeof(voxel_type))) {
		std::cout << "Reading .raw file failed   " << __LINE__ << std::endl;
		throw std::out_of_range("Reading .raw file error\n");

	}
	return true;
}

//Description
//find all the leaves of the octree and
//return the height of the octree
//

int min_width = 99999999, min_depth = 99999999, min_height = 99999999;

int find_leaves(octree_node * root, std::vector<octree_node *> & result, bool assignid = false) {
	if (root == nullptr) return 0;
	bool leaf = true;
	int max_height = -1;
	for (int i = 0; i < 8; i++) {
		max_height = std::max(find_leaves(root->children[i], result, assignid), max_height);
		if (root->children[i] != nullptr)
			leaf = false;
	}
	if (leaf == true && root->valid == true) {
		//if (root->valid == true)std::cout << "TRUE";
		//else std::cout << "FALSE";
		//std::cout << " " << root->min_point << " " << root->max_point << " " << root->min_data_point << " " << root->max_data_point << std::endl;
#ifdef LEAF_CHECK
		assert(root->min_point < root->max_point);
		assert(root->min_point == root->min_data_point);
		assert(root->max_point >= root->max_data_point);
#endif
		if (assignid == true)root->id = new_id();
		//Find minimum blocks
		if ((min_width > root->max_point.x - root->min_point.x)
			|| (min_depth > root->max_point.y - root->min_point.y)
			|| (min_height > root->max_point.z - root->min_point.z)) {
			min_width = root->max_point.x - root->min_point.x;
			min_depth = root->max_point.y - root->min_point.y;
			min_height = root->max_point.z - root->min_point.z;
		}
		result.push_back(root);
	}
	return max_height + 1;
}
//
//extend the octree as a complete octree,invalid leaf node excluded
//

void extend_octree(octree_node * root, int current_height, int max_height) {
	//达到最大高度
	if (current_height >= max_height || root == nullptr || root->valid == false)return;
	else if (is_leaf(root) == false) {	//如果不是叶节点，递归的对每个子节点进行扩展
		for (int i = 0; i < 8; i++) {
			extend_octree(root->children[i], current_height + 1, max_height);
		}
	}
	else {
		//如果是叶节点，扩展叶节点，id和父节点相同
		octree_node * new_node;
		pos_type dx = (root->max_point.x - root->min_point.x);
		pos_type dy = (root->max_point.y - root->min_point.y);
		pos_type dz = (root->max_point.z - root->min_point.z);
		pos_type dx1 = dx / 2, dx2;
		pos_type dy1 = dy / 2, dy2;
		pos_type dz1 = dz / 2, dz2;
		if (dx % 2 == 1)dx2 = dx1 + 1; else dx2 = dx1;
		if (dy % 2 == 1)dy2 = dy1 + 1; else dy2 = dy1;
		if (dz % 2 == 1)dz2 = dz1 + 1; else dz2 = dz1;
		point3d offset[8][2] = { { point3d(0, 0, 0), point3d(dx1, dy1, dz1) },
		{ point3d(dx1, 0, 0), point3d(dx1 + dx2, dy1, dz1) },
		{ point3d(0, dy1, 0), point3d(dx1, dy1 + dy2, dz1) },
		{ point3d(dx1, dy1, 0), point3d(dx1 + dx2, dy1 + dy2, dz1) },
		{ point3d(0, 0, dz1), point3d(dx1, dy1, dz1 + dz2) },
		{ point3d(dx1, 0, dz1), point3d(dx1 + dx2, dy1, dz1 + dz2) },
		{ point3d(0, dy1, dz1), point3d(dx1, dy1 + dy2, dz1 + dz2) },
		{ point3d(dx1, dy1, dz1), point3d(dx1 + dx2, dy1 + dy2, dz1 + dz2) } };

		//children node distribute first along x axis,and then y axis ...
		for (int z = 0; z < 2; z++) {
			for (int y = 0; y < 2; y++) {
				for (int x = 0; x < 2; x++) {
					int i = x + y * 2 + z * 4;
					new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1], root->id);
					new_node->parent = root;
					octree_node *& ref = root->children[x + y * 2 + 4 * z];
					ref = new_node;
					extend_octree(ref, current_height + 1, max_height);
				}
			}
		}
	}
}
void extend_regular_octree(octree_node * root, int current_height, int max_height) {
	//达到最大高度
	if (current_height >= max_height || root == nullptr || root->valid == false)return;
	else if (is_leaf(root) == false) {	//如果不是叶节点，递归的对每个子节点进行扩展
		for (int i = 0; i < 8; i++) {
			extend_regular_octree(root->children[i], current_height + 1, max_height);
		}
	}
	else {
		assert(root->id != -1);
		//如果是叶节点，扩展叶节点，id和父节点相同
		octree_node * new_node;
		pos_type dx = (root->max_point.x - root->min_point.x);
		pos_type dy = (root->max_point.y - root->min_point.y);
		pos_type dz = (root->max_point.z - root->min_point.z);
		assert(dx % 2 == 0);
		assert(dy % 2 == 0);
		assert(dz % 2 == 0);
		pos_type dx1 = dx / 2;
		pos_type dy1 = dy / 2;
		pos_type dz1 = dz / 2;
		pos_type dx2 = dx1;
		pos_type dy2 = dy1;
		pos_type dz2 = dz1;

		//size of octree
		point3d offset[8][2] = { { point3d(0, 0, 0), point3d(dx1, dy1, dz1) },
		{ point3d(dx1, 0, 0), point3d(dx1 + dx2, dy1, dz1) },
		{ point3d(0, dy1, 0), point3d(dx1, dy1 + dy2, dz1) },
		{ point3d(dx1, dy1, 0), point3d(dx1 + dx2, dy1 + dy2, dz1) },
		{ point3d(0, 0, dz1), point3d(dx1, dy1, dz1 + dz2) },
		{ point3d(dx1, 0, dz1), point3d(dx1 + dx2, dy1, dz1 + dz2) },
		{ point3d(0, dy1, dz1), point3d(dx1, dy1 + dy2, dz1 + dz2) },
		{ point3d(dx1, dy1, dz1), point3d(dx1 + dx2, dy1 + dy2, dz1 + dz2) } };

		//size of volume data
		bool subdivide[8] = { true, true, true, true, true, true, true, true };
		pos_type d_dx = (root->max_data_point.x - root->min_data_point.x);
		pos_type d_dy = (root->max_data_point.y - root->min_data_point.y);
		pos_type d_dz = (root->max_data_point.z - root->min_data_point.z);

		pos_type d_dx1 = std::min(d_dx, dx1);
		pos_type d_dy1 = std::min(d_dy, dy1);
		pos_type d_dz1 = std::min(d_dz, dz1);
		pos_type d_dx2 = std::max(d_dx - d_dx1, 0);
		pos_type d_dy2 = std::max(d_dy - d_dy1, 0);
		pos_type d_dz2 = std::max(d_dz - d_dz1, 0);
		if (d_dx2 == 0) {
			subdivide[1] = false;
			subdivide[3] = false;
			subdivide[5] = false;
			subdivide[7] = false;
		}
		if (d_dy2 == 0) {
			subdivide[2] = false;
			subdivide[3] = false;
			subdivide[6] = false;
			subdivide[7] = false;
		}
		if (d_dz2 == 0) {
			subdivide[4] = false;
			subdivide[5] = false;
			subdivide[6] = false;
			subdivide[7] = false;
		}

		point3d data_offset[8][2] = { { point3d(0, 0, 0), point3d(d_dx1, d_dy1, d_dz1) },
		{ point3d(d_dx1, 0, 0), point3d(d_dx1 + d_dx2, d_dy1, d_dz1) },
		{ point3d(0, d_dy1, 0), point3d(d_dx1, d_dy1 + d_dy2, d_dz1) },
		{ point3d(d_dx1, d_dy1, 0), point3d(d_dx1 + d_dx2, d_dy1 + d_dy2, d_dz1) },
		{ point3d(0, 0, d_dz1), point3d(d_dx1, d_dy1, d_dz1 + d_dz2) },
		{ point3d(d_dx1, 0, d_dz1), point3d(d_dx1 + d_dx2, d_dy1, d_dz1 + d_dz2) },
		{ point3d(0, d_dy1, d_dz1), point3d(d_dx1, d_dy1 + d_dy2, d_dz1 + d_dz2) },
		{ point3d(d_dx1, d_dy1, d_dz1), point3d(d_dx1 + d_dx2, d_dy1 + d_dy2, d_dz1 + d_dz2) } };


		//children node distribute first along x axis,and then y axis ...
		for (int z = 0; z < 2; z++) {
			for (int y = 0; y < 2; y++) {
				for (int x = 0; x < 2; x++) {
					int i = x + y * 2 + z * 4;
					if (subdivide[i] == true) {
						new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1], root->min_data_point + data_offset[i][0], root->min_data_point + data_offset[i][1], subdivide[i], root->id);
					}
					else {
						new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1], root->min_point + offset[i][0], root->min_point + offset[i][0], subdivide[i], root->id);
					}
					new_node->parent = root;
					octree_node *& ref = root->children[x + y * 2 + 4 * z];
					ref = new_node;
					extend_regular_octree(ref, current_height + 1, max_height);
				}
			}
		}
	}
}

//Delete octree
void destroy_octree(octree_node * root) {
	if (root == nullptr)return;
	if (is_leaf(root) == true) {
		delete root;
	}
	else {
		for (int i = 0; i < 8; i++) {
			destroy_octree(root->children[i]);
		}
	}
}
void min_max_ent(const octree_node *root, double & min_ent, double & max_ent) {
	if (root == nullptr)return;
	if (is_leaf(root) == true) {		//leaf node
		min_ent = std::min(min_ent, root->ent);
		max_ent = std::max(max_ent, root->ent);
	}
	else {
		for (int i = 0; i < 8; i++) {
			min_max_ent(root->children[i], min_ent, max_ent);
		}
	}
}

int estimate_sgmmcluster_size(const octree_node * root, const double & min_ent, const double & max_ent) {
	int cur_size = 0;
	if (root == nullptr)return cur_size;
	if (is_leaf(root) == true) {		//叶节点直接估计大小就可以了
		double delta_ent = (max_ent - min_ent) / (max_cluster_num - min_cluster_num);
		int cluster_num = (root->ent - min_ent) / delta_ent + min_cluster_num;
		if (cluster_num > max_cluster_num)
			cluster_num = max_cluster_num;
		//estimate a single block size
		cur_size += sizeof(unsigned char);	//cluster_num
		cur_size += sizeof(float);			//prior probility
		cur_size += sizeof(unsigned char); // guass count
		cur_size += sizeof(float);		//sample value


		/*Generally, there are 4 gaussian in a GMM
		for one gaussion,there are 1- float weight,
		1-float determinate ,3-float to store mean and
		6-float covmatrix for 3d gaussian*/
		cur_size += 4 * (sizeof(float) + sizeof(float) + 3 * sizeof(float) + 6 * sizeof(float));
		return cluster_num*cur_size;
	}
	else {
		for (int i = 0; i < 8; i++) {
			cur_size += estimate_sgmmcluster_size(root->children[i], min_ent,max_ent);
		}
		return cur_size;
	}
}

bool save_leaves_text(const std::string & path, const std::string & file_name, const std::vector<octree_node*> & leaves, std::size_t width, std::size_t depth, std::size_t height) {
	std::size_t total_size = width*depth*height;
	id_type * id_table = new int[total_size];

	std::ofstream out_file(path + file_name + ".oc", std::ios::binary);
	if (out_file.is_open() == false) {
		return false;
	}
	int id = 0;
	for (const auto item : leaves) {
		auto xmin = item->min_point.x;
		auto ymin = item->min_point.y;
		auto zmin = item->min_point.z;
		auto xmax = item->max_point.x;
		auto ymax = item->max_point.y;
		auto zmax = item->max_point.z;
		for (auto z = zmin; z < zmax; z++) {
			for (auto y = ymin; y < ymax; y++) {
				for (auto x = xmin; x < xmax; x++) {
					pos_type pos = x + y*width + z*width*depth;
					id_table[pos] = item->id;
				}
			}
		}
		out_file << xmin << " " << ymin << " " << zmin << " " << xmax << " " << ymax << " " << zmax << " " << item->id << std::endl;
		//id++;
	}
	out_file.close();
	std::ofstream out_table_file(path + file_name + ".idt");
	if (out_table_file.is_open() == false) {
		return false;
	}
	for (int i = 0; i < total_size; i++) {
		out_table_file << id_table[i] << std::endl;
	}
	//out_table_file.write((const char *)id_table,sizeof(id_type)*total_size);
	out_table_file.close();
	delete[] id_table;
	return true;
}

bool save_exleaves_text(const std::string & path, const std::string & file_name, const std::vector<octree_node*> & leaves, std::size_t width, std::size_t depth, std::size_t height) {
	std::size_t total_size = width*depth*height;
	id_type * id_table = new int[total_size];

	std::ofstream out_file(path + file_name + ".ocex", std::ios::binary);
	if (out_file.is_open() == false) {
		return false;
	}
	int count = 0;
	for (const auto item : leaves) {
		auto xmin = item->min_point.x;
		auto ymin = item->min_point.y;
		auto zmin = item->min_point.z;
		auto xmax = item->max_point.x;
		auto ymax = item->max_point.y;
		auto zmax = item->max_point.z;
		for (auto z = zmin; z < zmax; z++) {
			for (auto y = ymin; y < ymax; y++) {
				for (auto x = xmin; x < xmax; x++) {
					pos_type pos = x + y*width + z*width*depth;
					id_table[pos] = item->id;
				}
			}
		}
		out_file <</* xmin << " " << ymin << " " << zmin << " " << xmax << " " << ymax << " " << zmax << " " << */item->id << std::endl;
		count++;
	}
	out_file.close();
	std::ofstream out_table_file(path + file_name + ".idtex");
	if (out_table_file.is_open() == false) {
		return false;
	}
	for (int i = 0; i < total_size; i++) {
		out_table_file << id_table[i] << std::endl;
	}
	//out_table_file.write((const char *)id_table,sizeof(id_type)*total_size);
	out_table_file.close();
	std::cout << "extended octree leaves:" << count << std::endl;
	delete[] id_table;
	return true;
}
//REGULAR OCTREE 


bool re_save_leaves_text(const std::string & path, const std::string & file_name, const std::vector<octree_node*> & leaves, std::size_t width, std::size_t depth, std::size_t height) {

	//assert(width == ::data_width);
	//assert(depth == ::data_depth);
	//assert(height == ::data_height);

	std::size_t total_size = width*depth*height;
	id_type * id_table = new int[total_size];

	std::ofstream out_file(path + file_name + ".reoc", std::ios::binary);
	if (out_file.is_open() == false) {
		return false;
	}

	std::ofstream out_file_cluster_num(path + file_name + ".noc");
	if (out_file_cluster_num.is_open() == false) {
		std::cout << "can't not create file for cluster num\n";
		return false;
	}

	out_file << leaves.size() << std::endl;
	int id = 0;
	int data_count = 0;
	double min_ent = 99999999;
	double max_ent = -1;
	for (const auto item : leaves) {
		if (item->valid == false) {
			std::cout << "error:" << __LINE__ << std::endl;
			exit(0);
		}
		auto xmin = item->min_point.x;
		auto ymin = item->min_point.y;
		auto zmin = item->min_point.z;
		auto xmax = item->max_point.x;
		auto ymax = item->max_point.y;
		auto zmax = item->max_point.z;
		auto dxmin = item->min_data_point.x;
		auto dymin = item->min_data_point.y;
		auto dzmin = item->min_data_point.z;
		auto dxmax = item->max_data_point.x;
		auto dymax = item->max_data_point.y;
		auto dzmax = item->max_data_point.z;
		assert(item->min_point == item->min_data_point);
		data_count += (dzmax - dzmin)*(dymax - dymin)*(dxmax - dxmin);
		for (auto z = dzmin; z < dzmax; z++) {
			for (auto y = dymin; y < dymax; y++) {
				for (auto x = dxmin; x < dxmax; x++) {
					pos_type pos = x + y*width + z*width*depth;
					id_table[pos] = item->id;
				}
			}
		}
		min_ent = std::min(min_ent, item->ent);
		max_ent = std::max(max_ent, item->ent);
#ifdef INPUT
		out_file << dxmin << " " << dymin << " " << dzmin << " " << dxmax << " " << dymax << " " << dzmax << " "/*<< xmin << " " << ymin << " " << zmin << " " << xmax << " " << ymax << " " << zmax << " " */ << item->id << std::endl;
#endif

		id++;
	}
	std::cout << "max entropy:" << max_ent << "\n" << "min entropy:" << min_ent << std::endl;
	std::cout << total_size << " " << data_count << std::endl;
	out_file.close();
	std::cout << "Bounding box file output finished\n";
	std::ofstream out_table_file(path + file_name + ".reidt");
	if (out_table_file.is_open() == false) {
		return false;
	}
#ifdef INPUT
	for (int i = 0; i < total_size; i++) {
		out_table_file << id_table[i] << std::endl;
	}
#endif
	assert(max_cluster_num != min_cluster_num);
	double delta_ent = (max_ent - min_ent) / (max_cluster_num - min_cluster_num);
	if (min_ent < max_ent) {
		for (const auto item : leaves) {
			int cluster_num = (item->ent - min_ent) / delta_ent + min_cluster_num;
			if (cluster_num > max_cluster_num)
				cluster_num = max_cluster_num;
			out_file_cluster_num << cluster_num << std::endl;
		}
	}

	out_file_cluster_num.close();
	//out_table_file.write((const char *)id_table,sizeof(id_type)*total_size);
	out_table_file.close();
	delete[] id_table;
	return true;
}



bool re_save_exleaves_text(const std::string & path,
	const std::string & file_name,
	const std::vector<octree_node*> & leaves,
	pos_type width,
	pos_type depth,
	pos_type height,
	pos_type min_width = 0,
	pos_type min_depth = 0,
	pos_type min_height = 0) {

	assert(width == ::data_width);
	assert(depth == ::data_depth);
	assert(height == ::data_height);

	std::size_t total_size = width*depth*height;
	//id_type * id_table = new int[total_size];

	std::ofstream out_file(path + file_name + ".reocex", std::ios::binary);
	if (out_file.is_open() == false) {
		return false;
	}
	std::ofstream out_file_bin(path + file_name + ".reocexbin", std::ios::binary);
	if (out_file_bin.is_open() == false) {
		return false;
	}

	int count = 0;
	out_file << leaves.size() << std::endl;
	size_t s = leaves.size();
	out_file_bin.write((const char *)&s, sizeof(leaves.size()));
	for (const auto item : leaves) {
		//auto xmin = item->min_point.x;
		//auto ymin = item->min_point.y;
		//auto zmin = item->min_point.z;
		auto xmax = item->max_point.x;
		auto ymax = item->max_point.y;
		auto zmax = item->max_point.z;
		auto dxmin = item->min_data_point.x;
		auto dymin = item->min_data_point.y;
		auto dzmin = item->min_data_point.z;
		auto dxmax = item->max_data_point.x;
		auto dymax = item->max_data_point.y;
		auto dzmax = item->max_data_point.z;
		count += (dzmax - dzmin)*(dymax - dymin)*(dxmax - dxmin);
		//for (auto z = dzmin; z < dzmax; z++) {
		//	for (auto y = dymin; y < dymax; y++) {
		//		for (auto x = dxmin; x < dxmax; x++) {
		//			pos_type pos = x + y*width + z*width*depth;
		//			id_table[pos] = item->id;
		//		}
		//	}
		//}
		out_file << item->max_point << " " << item->id << std::endl;
		/*out_file_bin.write((const char *)&item->min_data_point.x, sizeof(item->min_data_point.x));
		out_file_bin.write((const char *)&item->min_data_point.y, sizeof(item->min_data_point.y));
		out_file_bin.write((const char *)&item->min_data_point.z, sizeof(item->min_data_point.z));
		out_file_bin.write((const char *)&item->max_data_point.x, sizeof(item->max_data_point.x));
		out_file_bin.write((const char *)&item->max_data_point.y, sizeof(item->max_data_point.y));
		out_file_bin.write((const char *)&item->max_data_point.z, sizeof(item->max_data_point.z));*/
	}
	out_file << min_width << " " << min_depth << " " << min_height << std::endl;
	out_file.close();

	out_file_bin.write((const char*)&min_width, sizeof(min_width));
	out_file_bin.write((const char*)&min_depth, sizeof(min_depth));
	out_file_bin.write((const char*)&min_height, sizeof(min_height));
	out_file_bin.close();
	//std::ofstream out_table_file(path + file_name + ".reidtex");
	//if (out_table_file.is_open() == false){
	//	return false;
	//}
	//for (int i = 0; i<total_size; i++){
	//	out_table_file << id_table[i] << std::endl;
	//}
	////out_table_file.write((const char *)id_table,sizeof(id_type)*total_size);
	//out_table_file.close();
	std::cout << "extended octree leaves:" << count << std::endl;

	//delete[] id_table;
	return true;
}
int count_leaf(const octree_node * root) {
	if (root == nullptr)return 0;
	if (is_leaf(root) == true) {
		return 1;
	}
	else {
		int leaves = 0;
		for (int i = 0; i < 8; i++) {
			leaves +=count_leaf(root->children[i]);
		}
		return leaves;
	}
}


int subdivision(int argc, char ** argv) {

	//input information about data
	std::string dir;
	std::cout << "--------------OCTREE MODULE-------------\n";
	std::cout << "input the data address\n";
	std::cin >> dir;

	std::string file_name;
	std::cout << "input the file name\n";
	std::cin >> file_name;

	int min_side;
	double ent_threshold;
	std::cout << "input width depth,height minside entry(5)\n";
	std::cin >> width >> depth >> height >> min_side >> ent_threshold;
	std::cout << "widtt:" << width << " depth:" << depth << " height:" << height << " min_side:" << min_side << " entropy:" << ent_threshold << std::endl;

	data_width = width;
	data_height = height;
	data_depth = depth;


	//
	width = next_pow_of_2(width);
	depth = next_pow_of_2(depth);
	height = next_pow_of_2(height);
	size_t max_side = std::max(std::max(width, depth), height);
	width = max_side;
	depth = max_side;
	height = max_side;

	std::cout << "next_pow_of_2:" << width << " " << depth << " " << height << std::endl;

	//Reading .raw file
	int total_size = data_width*data_depth*data_height;
	std::cout << dir + file_name + ".raw" << std::endl;
	volume_data = new unsigned char[total_size];
	if (read_data(dir + file_name + ".raw", volume_data, data_width, data_depth, data_height) == false) {
		std::cout << "can not open data file\n";
		exit(1);
	}

	//OCTree root node
	octree_node * root = nullptr;
	std::vector<octree_node *> leaves;
	std::vector<octree_node *> extended_octree_leaves;


	//a criterion for recursively creating a octree
	//entropy ent(volume_data, width, depth, height, ent_threshold);


	//create_regular_octree(volume_data, root, min_side, ent);



	//在这里要循环二分搜索看看是不是满足所给大小的条件
	int eps_size = 1 * 1024 * 1024;
	double c = 0.1;
	int cur_size = c*data_width*data_depth*data_height, estimate_size;

	double left_ent = 0.01;
	double right_ent = 10;
	ent_threshold = 5;
	int iterations = 0;
	std::ofstream time_consume(dir + file_name + "_OCTREETIME.txt");
	unsigned int begin_time = clock();
	while (true) {
		iterations++;
		entropy ent(volume_data, width, depth, height, ent_threshold);
		create_regular_octree(volume_data, root, min_side, ent);
		double min_ent = 999999999;
		double max_ent = -1;
		min_max_ent(root, min_ent, max_ent);
		estimate_size = estimate_sgmmcluster_size(root, min_ent, max_ent);
		std::cout << "----------\n";
	    std::wcout << "Estimate size:" << 1.0*estimate_size/1024.0/1024.0 <<"M."<<std::endl;
		std::cout << "Desired size:" << 1.0*cur_size / 1024.0 / 1024.0 << std::endl;
		std::cout << "Current Entropy:" << ent_threshold << std::endl;
		std::cout << "Number of leaf:" << count_leaf(root) << std::endl;
		if (std::abs(cur_size - estimate_size) < eps_size)//satisfied the expected size
			break;
		if (estimate_size - cur_size > eps_size) {
			//Increase the entropy so as to decrese the number of block
			left_ent = ent_threshold;
			ent_threshold = (right_ent + ent_threshold) / 2;
			if (iterations > 30) {
				ent_threshold += 0.001;
				if (ent_threshold > right_ent) {
					ent_threshold = right_ent;
					break;
				}
			}
		}
		else {
			right_ent = ent_threshold;
			ent_threshold = (left_ent + ent_threshold) / 2;
			if (iterations > 30) {
				ent_threshold -= 0.001;
				if (ent_threshold < left_ent) {
					ent_threshold = left_ent;
					break;
				}
			}
		}

		std::cout << "left:"<<left_ent<<" "<<right_ent<<" "<<"Next entropy:" << ent_threshold << std::endl;
		destroy_octree(root);
		root = nullptr;
	}
	unsigned int end_time = std::clock();
	std::cout << "Parition time:" << 1.0*(end_time - begin_time) / CLOCKS_PER_SEC<<"s." << std::endl;
	time_consume << "Parition time:" << 1.0*(end_time - begin_time) / CLOCKS_PER_SEC<<"s." << std::endl;
	time_consume.close();
	
	//std::wcout << "Estimate size:" << 1.0*estimate_size/1024.0/1024.0 << std::endl;
	std::cout << "=====================\n";
	std::wcout << "Iterations:" << iterations << std::endl;
	
	//collect leaves from octree and return the height of the tree
	int h = find_leaves(root, leaves, true);
	std::cout << "Height of the OCTree:" << h << std::endl;

	//checking the voxel from leaves whether it is equal to the total voxel
	int voxel_count = 0;
	std::cout << "Leaves in the OCTree:" << leaves.size() << std::endl;

	for (const auto ptr : leaves) {
		voxel_count += (ptr->max_data_point.x - ptr->min_data_point.x)*
			(ptr->max_data_point.y - ptr->min_data_point.y)*
			(ptr->max_data_point.z - ptr->min_data_point.z);
	}
	std::cout << "Voxel:" << voxel_count << std::endl;

	//save .reoc file 
	re_save_leaves_text(dir, file_name, leaves, data_width, data_depth, data_height);


	//Extending octree and exclude the empty leaf node
	std::cout << "Extending octree ...\n";
	extend_regular_octree(root, 1, h);

#ifdef LEAF_ID_CONTINUOUS_CHECK
	{
		//check leaf node if it is continuous
		int n_leaf = leaves.size();
		std::vector<int> count(n_leaf);
		std::vector<int> checks(n_leaf);
		for (auto const item : leaves) {
			id_type id = item->id;
			if (id >= n_leaf)std::cout << "id out of range\n";
			count[id]++;
		}
		bool continuous = true;
		for (int i = 0; i < n_leaf; i++) {
			if (count[i] != 1) {
				std::cout << "leaf node error\n";
			}
			if (count[i] == 0)continuous = false;
		}
		if (continuous == false)std::cout << "leaf node is not continuous\n";

		find_leaves(root, extended_octree_leaves);
		continuous = true;
		for (auto const item : extended_octree_leaves) {
			id_type id = item->id;
			if (id >= n_leaf)std::cout << "id is out of range\n";
			checks[id]++;
		}
		for (int i = 0; i < n_leaf; i++) {
			if (checks[i] == 0) {
				std::cout << "exleaf node error\n";
				continuous = false;
			}
		}
		if (continuous == false)std::cout << "exleaf node is not continuous\n";
	}
#endif
	std::cout << "Number of octree leaf node:" << leaves.size() << std::endl;
	std::cout << " Number of extended octree leaf node:" << extended_octree_leaves.size() << std::endl;

	//Save extended octree leaves node for real time rendering
	std::sort(extended_octree_leaves.begin(), extended_octree_leaves.end(), leaf_cmp);
	re_save_exleaves_text(dir, file_name, extended_octree_leaves, data_width, data_depth, data_height, min_width, min_depth, min_height);
	std::cout << "min side:" << min_width << " " << min_depth << " " << min_height << std::endl;
	std::cin.get();


	delete[] volume_data;
	destroy_octree(root);

	return 0;
}