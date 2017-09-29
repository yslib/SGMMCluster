#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>


static std::string disk_address = "d:/train/";


//int total_size = width * depth * height;

double RMSE(const std::string & src_address_1,
	const std::string & src_address_2,
	int width, int depth, int height) {
	int total_size = width * depth * height;
	unsigned char* src_1 = new unsigned char[total_size];
	unsigned char* src_2 = new unsigned char[total_size];

	std::ifstream f_src(src_address_1, std::ios::binary);
	if (f_src.is_open() == false) {
		//std::cout << "cant not open " << src_address_1 << std::endl;
		return -1;
	}
	f_src.read((char*)src_1, total_size * sizeof(unsigned char));
	f_src.close();

	f_src.open(src_address_2, std::ios::binary);
	if (f_src.is_open() == false) {
		//std::cout << "can not open " << src_address_2 << std::endl;
		return -1;
	}
	f_src.read((char*)src_2, total_size * sizeof(unsigned char));
	f_src.close();

	double rmse = 0.0;
	for (int i = 0; i < total_size; i++) {
		double distance = src_1[i] - src_2[i];
		rmse += distance * distance;
	}
	rmse /= total_size;
	rmse = sqrt(rmse);
	//std::cout << rmse << std::endl;
	delete src_1;
	delete src_2;
	return rmse;
}
int rmse(int argc,char ** argv) {
	int width = 480; // ÎïÌå¿í
	int depth = 720; // ÎïÌåÉî
	int height = 112;  // ÎïÌå¸ß
	std::cout << "input the .info file\n";
	std::string data_name;
	std::cin >> data_name;
	std::ifstream in_file(disk_address + data_name + ".info");
	if (in_file.is_open() == false) {
		std::cout << ".info file can not be opened\n";
		return 0;
	}

	in_file >> width >> depth >> height;
	//std::cout << width << " " << depth << " " << height << std::endl;
	std::string source_data_name;

	in_file >> source_data_name;
	std::string src_address_1 = disk_address + source_data_name;
	//std::cout << source_data_name << std::endl;
	std::string gmm_file;
	std::string sgmm_file;
	std::string sgmm_cluster_file;
	in_file >> gmm_file >> sgmm_file >> sgmm_cluster_file;
	//std::cout << gmm_file << " " << sgmm_file <<" "<< sgmm_cluster_file << std::endl;

	double r1 = RMSE(src_address_1, disk_address + gmm_file, width, depth, height);
	double r2 = RMSE(src_address_1, disk_address + sgmm_file, width, depth, height);
	double r3 = RMSE(src_address_1, disk_address + sgmm_cluster_file, width, depth, height);
	std::cout << data_name << ":" << std::endl;
	if (r1 != -1)std::printf("RMSE between RAW and GMM = \t%.2lf\n", r1);
	if (r2 != -1)std::printf("RMSE between RAW and SGMM = \t%.2lf\n", r2);
	if (r3 != -1)std::printf("RMSE between RAW and SGMMWidthCluster = \t%.2lf\n", r3);
	return 0;

}