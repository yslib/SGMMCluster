#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
const std::string addr_prefix = "d:/train/";
int get_file_size(const std::string & path) {
	std::ifstream in_file(path, std::ios::binary);
	int size;
	if (in_file.is_open() == true) {
		in_file.seekg(0, std::ios::end);
		size = static_cast<int>(in_file.tellg());
	}
	else {
		size = 0;
	}
	return size;
}
int size_of_source_file(const std::string & name) {
	std::string source_file = addr_prefix + name + ".raw";
	return get_file_size(source_file);
}
int size_of_sgmm_clusters(const std::string& name) {
	std::string sgmm_file = addr_prefix + name + "_SGMM_Cluster_Result.sgmm";
	int size1 = get_file_size(sgmm_file);
	std::string integrations_file = addr_prefix + name + "_integrations_sgmm_cluster";
	int size2 = get_file_size(integrations_file);
	//std::cout << size1 << " " << size2 << std::endl;
	return size1 + size2;
}
int size_of_sgmm(const std::string & name) {
	std::string sgmm_file = addr_prefix + name + "_SGMM_Result.sgmm";
	int size1 = get_file_size(sgmm_file);
	std::string integration_file = addr_prefix + name + "_integrations_sgmm";
	int size2 = get_file_size(integration_file);
	return size1 + size2;
}
int size_of_gmm(const std::string & name) {
	std::string gmm_file = addr_prefix + name + "_GMM_Result.gmm";
	return get_file_size(gmm_file);
}
int compress_ratio(int argc,char ** argv)
{
	std::string data_name;
	std::cout << "input the data name would be calculate\n";
	std::cin >> data_name;
	double source_file_size = size_of_source_file(data_name);
	if (source_file_size == 0) {
		std::cout << "can not open the source file or the size of source file is 0\n";
		return 0;
	}
	double block_gmm_size = size_of_gmm(data_name);
	double sgmm_size = size_of_sgmm(data_name);
	double sgmm_cluster_size = size_of_sgmm_clusters(data_name);
	double r1 = block_gmm_size / source_file_size;
	double r2 = sgmm_size / source_file_size;
	double r3 = sgmm_cluster_size / source_file_size;
	if (r1 != 0)printf("BlockGMM/SourceFile = %.4lf (%.2lf%%)\n", r1, r1 * 100);
	if (r2 != 0)printf("SGMM/SourceFile = %.4lf (%.2lf%%)\n", r2, r2 * 100);
	if (r3 != 0)printf("SGMMWithCluster/SourceFile = %.4lf (%.2lf%%)\n", r3, r3 * 100);
	return 0;
}