#pragma once
#ifndef _COMMANDS_H_
#define _COMMANDS_H_

/*all custom command*/
int cuda_device_query(int argc, char **argv);

int subdivision(int argc, char ** argv);

int txt2binarySGMMCluster(int argc, char ** argv);

int restoreRawBySGMMCluster(int argc, char ** argv);

int test_case_generate(int argc, char ** argv);

int compress_ratio(int argc, char ** argv);

int rmse(int argc, char ** argv);

int restore_raw_by_sgmm(int argc, char ** argv);

int restore_raw_by_gmm(int argc, char ** argv);

int txt2binarysgmm(int argc, char ** argv);

int txt2binarygmm(int argc, char ** argv);

int train_sgmm_cluster_octree(int argc, char **argv);

int train_sgmm(int gc, char ** argv);

int train_block_gmm(int argc, char **argv);

int draw_bounding_box(int argc, char ** argv);

int create_AABB_file(int argc, char ** argv);

int RealTimeVolumeRender(int argc, char ** argv);

int float2byte(int argc, char ** argv);

#endif/*_COMMADS_H_*/
