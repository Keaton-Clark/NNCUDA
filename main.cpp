// Include C++ header files.
#include <iostream>
#include <byteswap.h>
#include <vector>
#include <fstream>

// Include local CUDA header files.
#include "include/nncuda.cuh"


nncuda::Array load_img_data(char *img_file) {
	union {
		uint32_t data[4];
		struct {
			uint32_t magic_number;
			uint32_t num_imgs;
			uint32_t num_rows;
			uint32_t num_cols;
		} meta;
	} img_head;
	FILE *fptr = fopen(img_file, "rb");
	if (!fptr) exit(EXIT_FAILURE);
	fread(&img_head, 16, 1, fptr);
	for (int i = 0; i < 4; i++) img_head.data[i] = bswap_32(img_head.data[i]);
	int x = 2000;//img_head.meta.num_imgs;
	int y = img_head.meta.num_cols * img_head.meta.num_rows;
	int size = x * y * sizeof(float);
	float *data = (float*)malloc(x * y * sizeof(float*));
	uint8_t *tmp = (uint8_t*)malloc(y * sizeof(uint8_t));
	for (int i = 0; i < x; i++) {
		fread(tmp, 1, y, fptr);
		for (int n = 0; n < y; n++) {
			data[n * x + i] = (float)(tmp[n]) / 784;
		}
	}
	fclose(fptr);
	free(tmp);
	auto output = nncuda::Array(data, x, y);
	free(data);
	return output;
}

nncuda::Array load_lbl_data(char *lbl_file) {
	union {
		uint32_t data[2];
		struct {
			uint32_t magic_number;
			uint32_t num_lbls;
		} meta;
	} lbl_head;
	FILE *fptr = fopen(lbl_file, "rb");
	if (!fptr) exit(EXIT_FAILURE);
	fread(&lbl_head, 8, 1, fptr);
	for (int i = 0; i < 2; i++) lbl_head.data[i] = bswap_32(lbl_head.data[i]);
	int x = 2000;
	int y = 10;
	int size = x * y * sizeof(float);
	float *data = (float*)calloc(x * y, sizeof(float));
	uint8_t *tmp = (uint8_t*)malloc(x * x);
	fread(tmp, 1, x, fptr);
	for (int i = 0; i < x; i++) {
		data[x * tmp[i] + i] = 1;
	}
	fclose(fptr);
	free(tmp);
	nncuda::Array output(data, x, y);
	free(data);
	return output;
}

int main() {
	nncuda::Network network(load_img_data("samples/train_images"), load_lbl_data("samples/train_labels"), 0.1);
	for (int i = 0; i < 10000; i++) {
		network.forward_prop();
		network.back_prop();
		network.update();
		if (i % 100 == 0)
			std::cout << network.accuracy() << std::endl;
	}
}
