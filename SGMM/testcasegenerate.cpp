#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
int test_case_generate(int argc,char ** argv) {
	std::cout << "------------TEST CASE GENERATION MODULE----------\n";

	int side = 15;
	int block_side = 5;
	int block_num_per_side = side / block_side;
	int block_num = block_num_per_side*block_num_per_side*block_num_per_side;
	int block_side2 = block_side*block_side;
	int block_side3 = block_side2*block_side;
	int side3 = side*side*side;
	//std::vector<unsigned char> volume_data(side3);
	unsigned char * volume_data = new unsigned char[side3];
	for (int i = 0; i < side3; i++) {
		volume_data[i] = 0;
	}
	std::vector<int> ids(side3);
	//std::cout << volume_data.size() << " " << ids.size() << std::endl;
	std::ofstream oc_file_out("test.oc");
	if (oc_file_out.is_open() == false) {
		std::cout << "can not open oc file\n";
		return 0;
	}
	for (int i = 0; i < block_num_per_side; i++) {
		for (int j = 0; j < block_num_per_side; j++) {
			for (int k = 0; k < block_num_per_side; k++) {
				int id = k + j*block_num_per_side + i*block_num_per_side*block_num_per_side;
				int bx = k*block_side;
				int by = j*block_side;
				int bz = i*block_side;
				for (int z = 0; z < block_side; z++) {
					for (int y = 0; y < block_side; y++) {
						for (int x = 0; x < block_side; x++) {
							int gx = bx + x;
							int gy = by + y;
							int gz = bz + z;
							int index = gx + gy*side + gz*side*side;
							int tmp;
							if (k == 0 && j == 0 && i == 0)tmp = 255;
							//tmp = 256.0 * ((1.0*id) / (1.0*block_num));
							else tmp = 0;
							volume_data[index] = (unsigned char)tmp;
							ids[index] = id;
						}
					}
				}
				int maxbx = bx + block_side;
				int maxby = by + block_side;
				int maxbz = bz + block_side;
				oc_file_out << bx << " " << by << " " << bz << " " << maxbx << " " << maxby << " " << maxbz << " " << id << std::endl;
			}
		}
	}
	//std::cout << std::endl;
	std::ofstream test_file_out("test.raw");
	if (test_file_out.is_open() == false) {
		std::cout << "can not open .raw file\n";
		return 0;
	}
	test_file_out.write((const char *)volume_data, side3);
	//for (int i = 0; i < side3; i++){
	//	test_file_out<< (int)volume_data[i] << " ";
	//	if (i%side == 0)std::cout << std::endl;
	//}
	test_file_out.close();

	std::ofstream idt_file_out("test.idt");
	if (idt_file_out.is_open() == false) {
		std::cout << "can not open idt file\n";
		return 0;
	}
	for (int i = 0; i < side3; i++) {
		idt_file_out << ids[i] << std::endl;
	}
	idt_file_out.close();
	return 0;
}