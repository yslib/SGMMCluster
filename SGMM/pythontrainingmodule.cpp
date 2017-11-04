#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#ifdef EMBEDDING_PYTHON2
#include <Python.h>
#endif

int train_sgmm(int gc, char ** argv) {
	std::cout << "--------------TRAIN SGMM MODULE-------------\n";
	std::string disk_address;
	std::cout << "input data address\n";
	std::cin >> disk_address;

	std::string file_name;
	std::cout << "input file name\n";
	std::cin >> file_name;

	std::string width;
	std::string depth;
	std::string height;
	std::string file_num;
	std::string side;
	std::cout << "width depth height side file_num\n";
	std::cin >> width >> depth >> height >>side>> file_num;

	std::string cmd{ "python C:/Users/ysl/Code/SGMM/SGMM/TrainSGMM.py"};
	cmd += " "+disk_address+" "+file_name+" " + width + " " + depth + " " + height + " " + side + " " + file_num;
	clock_t tbegin = clock();
	system(cmd.c_str());
	clock_t tend = clock();
	int t = (tend - tbegin) / CLOCKS_PER_SEC;
	std::cout << "Trainning time:" << t<< "s\n";
	std::ofstream time_file(disk_address + file_name + "sgmm" + ".tm");
	if (time_file.is_open() == true) {
		time_file << "SGMM Training Time:" << t << std::endl;
	}
	time_file.close();
	std::cin.get();
	return 0;
}
int train_block_gmm(int argc, char **argv) {
	std::cout << "---------------TRAIN BLOCK GMM MODUEL-------------\n";
	std::string disk_address;
	std::cout << "input data address\n";
	std::cin >> disk_address;

	std::string file_name;
	std::cout << "input file name\n";
	std::cin >> file_name;

	std::string width;
	std::string depth;
	std::string height;
	std::string file_num;
	std::string side;
	std::cout << "width depth height side file_num\n";
	std::cin >> width >> depth >> height >> side >> file_num;

	std::string cmd{ "python C:/Users/ysl/Code/SGMM/SGMM/TrainBlockGMM.py" };
	cmd += " " + disk_address + " " + file_name + " " + width + " " + depth + " " + height + " " + side + " " + file_num;
	clock_t tbegin = clock();
	system(cmd.c_str());
	clock_t tend = clock();
	int t = (tend - tbegin) / CLOCKS_PER_SEC;
	std::cout << "Trainning time:" << t << "s\n";
	std::ofstream time_file(disk_address + file_name + "gmm" + ".tm");
	if (time_file.is_open() == true) {
		time_file << "GMM Training Time:" << t << std::endl;
	}
	time_file.close();
	std::cin.get();
	return 0;
}

int train_sgmm_cluster_octree(int gc,char **gv)
{
	std::cout << "---------------TRAIN SGMM CLUSTER MODULE-------------\n";
	std::string python_home;
	std::string python_script_path;

	std::string disk_address;
	std::cout << "input data address\n";
	std::cin >> disk_address;

	std::string file_name;
	std::cout << "input file name\n";
	std::cin >> file_name;

	std::string width;
	std::string depth;
	std::string height;
	std::string file_num;
	std::cout << "width depth height file_num\n";
	std::cin >> width >> depth >> height >> file_num;

#ifdef EMBEDDING_PYTHON2


	int argc = 5;
	const char *argv[5];

	argv[0] = file_name.c_str();
	argv[1] = width.c_str();
	argv[2] = depth.c_str();
	argv[3] = height.c_str();
	argv[4] = file_num.c_str();
	
	//Set python runtime enviroments
	Py_SetPythonHome("C:/Users/ysl/Anaconda2");

	//Initialize python 
	Py_Initialize();
	if (!Py_IsInitialized()) {
		std::cout << "python initailzed failed\n";
		return 0;
	}

	// send console parameters to python
	PySys_SetArgv(argc, const_cast<char**>(argv));

	if (PyRun_SimpleString("execfile('../SGMM/testforcppcall.py')")) {
		std::cout << "failed to execute python script\n";
		return 0;
	}

	Py_Finalize();
#endif

	std::string cmd{ "python C:/Users/ysl/Code/SGMM/SGMM/TrainSGMMWithClusterOCTree.py" };
	cmd += +" "+disk_address + " " + file_name + " " + width + " " + depth + " " + height + " " + file_num;
	clock_t tbegin = clock();
	std::system(cmd.c_str());
	clock_t tend = clock();
	int t = (tend - tbegin) / CLOCKS_PER_SEC;
	std::cout << "Trainning time:" << t << "s\n";
	std::ofstream time_file(disk_address + file_name + "sgmmcluster" + ".tm");
	if (time_file.is_open() == true) {
		time_file << "SGMMCluster Training Time:" << t << std::endl;
	}
	time_file.close();
	std::cin.get();
	return 0;
}
