#ifndef NNCUDA_CUH
#define NNCUDA_CUH

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <iostream>


typedef struct {
	uint32_t x;
	uint32_t y;
	uint32_t size;
	float *data;
} nncuda_array;

typedef struct {
	nncuda_array *input;
	nncuda_array *labels;
	nncuda_array *W[2];
	nncuda_array *b[2];
	nncuda_array *Z[2];
	nncuda_array *A[2];
	nncuda_array *dZ[2];
} nncuda_network;

void nncuda_print_array(nncuda_array *array);
void nncuda_print_array_transpose(nncuda_array *array);
nncuda_network *nncuda_init(nncuda_array *input, nncuda_array *labels);
void nncuda_forward_prop(nncuda_network *network);
void nncuda_back_prop(nncuda_network *network);
void nncuda_free_array(nncuda_array *array);
void nncuda_update_layers(float alpha);
nncuda_array *nncuda_zero_array(uint32_t x, uint32_t y);
nncuda_array *nncuda_rand_array(uint32_t x, uint32_t y);
bool nncuda_realloc_array(nncuda_array *array, uint32_t x, uint32_t y);

#endif //NNCUDA_CUH
