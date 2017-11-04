#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include "funcs.h"


std::size_t size_of_source_file(const std::string & path,const std::string & name) {
	std::string source_file = path + name + ".raw";
	return get_file_size(source_file);
}
std::size_t size_of_sgmm_clusters(const std::string & path,const std::string& name) {
	std::string sgmm_file = path+ name + "_SGMM_Cluster_Result.sgmm";
	std::size_t size1 = get_file_size(sgmm_file);
	std::string integrations_file = path + name + "_integrations_sgmm_cluster";
	std::size_t size2 = 0;

	//std::size_t size3 = get_file_size(path + name + ".reocex");
	std::size_t size3 = get_file_size(path + name + ".reocexbin");
	//std::cout << size1 << " " << size2 << std::endl;
	return size1 + size2 + size3;
}
std::size_t size_of_sgmm(const std::string & path,const std::string & name) {
	std::string sgmm_file = path + name + "_SGMM_Result.sgmm";
	std::size_t size1 = get_file_size(sgmm_file);
	std::string integration_file = path + name + "_integrations_sgmm";
	std::size_t size2 = 0;
	return size1 + size2;
}
std::size_t size_of_gmm(const std::string & path,const std::string & name) {
	std::string gmm_file = path + name + "_GMM_Result.gmm";
	return get_file_size(gmm_file);
}
int compress_ratio(int argc,char ** argv)
{
	std::cout << "---------------COMPRESS RATIO MODULE--------------\n";
	std::string data_name;
	std::string data_address;
	std::cout << "input data address\n";
	std::cin >> data_address;
	std::cout << "input the data name would be calculate\n";
	std::cin >> data_name;
	double source_file_size = size_of_source_file(data_address,data_name);
	if (source_file_size == 0) {
		std::cout << "can not open the source file or the size of source file is 0\n";
		return 0;
	}
	double block_gmm_size = size_of_gmm(data_address,data_name);
	double sgmm_size = size_of_sgmm(data_address,data_name);
	double sgmm_cluster_size = size_of_sgmm_clusters(data_address,data_name);
	double r1 = block_gmm_size / source_file_size;
	double r2 = sgmm_size / source_file_size;
	double r3 = sgmm_cluster_size / source_file_size;
	std::cout << "Data:" << data_name << std::endl;
	if (r1 != 0)printf("BlockGMM/SourceFile = %.4lf (%.2lf%%)\n", r1, r1 * 100);
	if (r2 != 0)printf("SGMM/SourceFile = %.4lf (%.2lf%%)\n", r2, r2 * 100);
	if (r3 != 0)printf("SGMMWithCluster/SourceFile = %.4lf (%.2lf%%)\n", r3, r3 * 100);
	std::cin.get();
	return 0;
}