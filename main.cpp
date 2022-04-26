// Include C++ header files.
#include <iostream>
#include <byteswap.h>

// Include local CUDA header files.
#include "include/nncuda.cuh"



nncuda_array *load_img_data(char *img_file) {
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
	nncuda_array *output = (nncuda_array*)malloc(sizeof(nncuda_array));
	output->x = 2000;//img_head.meta.num_imgs;
	output->y = img_head.meta.num_cols * img_head.meta.num_rows;
	output->size = output->x * output->y * sizeof(float);
	output->data = (float*)malloc(output->x * output->y * sizeof(float*));
	uint8_t *tmp = (uint8_t*)malloc(output->y * sizeof(uint8_t));
	for (int i = 0; i < output->x; i++) {
		fread(tmp, 1, output->y, fptr);
		for (int n = 0; n < output->y; n++) {
			output->data[n * output->x + i] = (float)(tmp[n]);
		}
	}
	fclose(fptr);
	free(tmp);
	return output;
}

nncuda_array *load_lbl_data(char *lbl_file) {
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
	nncuda_array *output = (nncuda_array*)malloc(sizeof(nncuda_array));
	output->x = 2000;
	output->y = 10;
	output->size = output->x * output->y * sizeof(float);
	output->data = (float*)calloc(output->x * output->y, sizeof(float));
	uint8_t *tmp = (uint8_t*)malloc(output->x * output->x);
	fread(tmp, 1, output->x, fptr);
	for (int i = 0; i < output->x; i++) {
		output->data[output->x * tmp[i] + i] = 1;
	}
	fclose(fptr);
	free(tmp);
	return output;
}

int main() {
	nncuda_array *labels = load_lbl_data("samples/train_labels");
	nncuda_array *input = load_img_data("samples/train_images");
	nncuda_network *network = nncuda_init(input, labels);
	nncuda_forward_prop(network);
}
