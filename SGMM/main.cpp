#include "menu.h"
#include "commands.h"
#include "funcs.h"
#include <sstream>
#include <fstream>

#include <iostream>
int test(int argc, char ** argv) {

	std::string path;
	std::string name;
	std::cin >> path >> name;
	int width, depth, height, side;
	std::cin >> width >> depth >> height >> side;
	std::vector<AABB> aabbs = create_regular_boundingbox(width, depth, height, side, side, side);
	create_AABB_file(path, name, aabbs);
	for (const AABB&aabb : aabbs) {
		std::cout << aabb.min_point << " " << aabb.max_point << std::endl;
	}

	std::cin.get();
	return 0;
}

//Entry Point
int main(int argc,char ** argv)
{
	//buit-in command
	MenuConfig("h", "Show all commands.",help);
	MenuConfig("q", "Quit from the program.", quit);
	MenuConfig("c", "Cuda Device Query", cuda_device_query);
	MenuConfig("t", "Test", test);
	MenuConfig("r", "Real Time Rendering", RealTimeVolumeRender);

	//command for sgmm cluster
	MenuConfig("1", "Subdivide the volume data by octree.", subdivision);
	MenuConfig("2", "Train sgmm cluster", train_sgmm_cluster_octree);
	MenuConfig("3", "Convert text file to binary file generated by SGMMCluster training", txt2binarySGMMCluster);
	MenuConfig("4", "Restore and calculate integrations.", restoreRawBySGMMCluster);

	//command for sgmm
	MenuConfig("5", "Train sgmm", train_sgmm);
	MenuConfig("6", "Convert text file to binary file generated by SGMM training.", txt2binarysgmm);
	MenuConfig("7", "Restore and calculate integrations for sgmm", restore_raw_by_sgmm);

	//command for block gmm
	MenuConfig("8", "Train block gmm", train_block_gmm);
	MenuConfig("9", "Convert text file to binary file generated by GMM training.", txt2binarygmm);
	MenuConfig("10", "Calculate the integrations for gmm", restore_raw_by_gmm);

	//other operations
	MenuConfig("11", "Calculate the compression ratio.", compress_ratio);
	MenuConfig("12", "Calculate the rmse between two volume data.", rmse);
	MenuConfig("13", "Generate test case", test_case_generate);
	MenuConfig("14", "Create Bounding box file for sgmm and block gmm", create_AABB_file);
	MenuConfig("15", "Draw bounding box for .raw file",draw_bounding_box);
	MenuConfig("16", "Convert float file to byte file", float2byte);
	//exec menu
	ExecuteMenu();
	return 0;
}