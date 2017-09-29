#include "menu.h"
#include "commands.h"
#include "structuresfunctions.h"
#include <sstream>
#include <fstream>

#include <iostream>
int test(int argc, char ** argv) {
	std::cout << "-----------------TEST MODULE-----------------\n";
	int width, depth, height, side;
	std::cin >> width >> depth >> height >> side;
	std::vector<AABB> aabbs = create_regular_boundingbox(width, depth, height, side, side, side);
	for (int i = 0; i < aabbs.size(); i++) {
		const point3d & a = aabbs[i].min_point;
		const point3d & b = aabbs[i].max_point;
		std::cout << a.x << " " << a.y << " " << a.z << " " << b.x << " " << b.y << " " << b.z << std::endl;
	}
	std::cin.get();
	return 0;
}

//Entry Point
int main(int argc,char ** argv)
{
	//buit-in command
	MenuConfig("h", "show all commands.",help);
	MenuConfig("q", "quit from the program.", quit);
	MenuConfig("c", "Cuda Device Query", cuda_device_query);

	//command for sgmm cluster
	MenuConfig("1", "Subdividing the volume data by octree.", subdivision);
	MenuConfig("2", "train sgmm cluster", train_sgmm_cluster_octree);
	MenuConfig("3", "convert text file to binary file generated by SGMMCluster training", txt2binarySGMMCluster);
	MenuConfig("4", "restore and calculate integrations.", restoreRawBySGMMCluster);

	//command for sgmm
	MenuConfig("5", "train sgmm", train_sgmm);
	MenuConfig("6", "convert text file to binary file generated by SGMM training.", txt2binarysgmm);
	MenuConfig("7", "restore and calculate integrations for sgmm", restore_raw_by_sgmm);

	//command for block gmm
	MenuConfig("8", "train block gmm", train_block_gmm);
	MenuConfig("9", "convert text file to binary file generated by GMM training.", txt2binarygmm);
	MenuConfig("10", "calculating the integrations for gmm", restore_raw_by_gmm);

	//other operations
	MenuConfig("11", "calculating the compression ratio.", compress_ratio);
	MenuConfig("12", "calculating the rmse between two volume data.", rmse);
	MenuConfig("13", "generate test case", test_case_generate);
	MenuConfig("14", "draw bounding box for .raw file",draw_bounding_box);

	//test command
	MenuConfig("t", "test", test);

	//exec menu
	ExecuteMenu();
	return 0;
}