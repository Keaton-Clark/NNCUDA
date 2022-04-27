#include "../include/nncuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

#define DIM_COMP(_ARRAY1, _ARRAY2) ((_ARRAY1->x == _ARRAY2->x) && (_ARRAY1->y ==  _ARRAY2->y))
#define ERR(_STR) fprintf(stderr, _STR); exit(EXIT_FAILURE)

#define BLOCKSIZE 32
#define GRIDSIZE(_ITERATIONS) (_ITERATIONS + (BLOCKSIZE - 1)) / BLOCKSIZE

void nncuda_add_array(nncuda_array *dest, nncuda_array *array1, nncuda_array *array2);
void nncuda_dot_array(nncuda_array *dest, nncuda_array *array1, nncuda_array *array2);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

nncuda_array *nncuda_zero_array(uint32_t x, uint32_t y) {
	nncuda_array *output = (nncuda_array*)calloc(1, sizeof(nncuda_array));
	output->x = x;
	output->y = y;
	output->size = x * y * sizeof(float);
	gpuErrchk( cudaMalloc((void**)&output->data, output->size) );
	float buff[x * y];
	memset(buff, 0, output->size);
	gpuErrchk( cudaMemcpy(output->data, buff, output->size, cudaMemcpyHostToDevice) );
	return output;
}

nncuda_array *nncuda_rand_array(uint32_t x, uint32_t y) {
	nncuda_array *output = (nncuda_array*)calloc(1, sizeof(nncuda_array));
	output->x = x;
	output->y = y;
	output->size = x * y * sizeof(float);
	gpuErrchk( cudaMalloc((void**)&output->data, output->size) );
	float buff[x * y];
	for (int i = 0; i < x * y; i++) {
		buff[i] = ((double)rand()/(double)(RAND_MAX)) - .5;
		//buff[i] = (float)(rand()%5);
	}
	gpuErrchk( cudaMemcpy(output->data, buff, output->size, cudaMemcpyHostToDevice) );
	return output;
}

void nncuda_print_array(nncuda_array* array) {
	float tmp[array->x * array->y];
	gpuErrchk( cudaMemcpy(tmp, array->data, array->size, cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("_______________________\n");
	for (int n = 0; n < array->x; n++) {
		for (int i = 0; i < array->y; i++) {
			printf("|%lf", tmp[(array->x * i) + n]);
		}
		printf("|\n");
	}
}

__global__ void nncuda_add_array_vector_kernel(float *dest, float *array1, float *array2, uint32_t count, uint32_t wA) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		dest[tid] = array1[tid] + array2[tid / wA];
}

void nncuda_add_array_vector(nncuda_array *dest, nncuda_array *array1, nncuda_array *array2) {
	if (array1->y != array2->y) {ERR("In add_array_vector, array1->y and array2->y are not equal\n");}
	if (!DIM_COMP(array1, dest)) {ERR("In add_array_vector, shape of dest is not equal to array1\n");}
	uint32_t n = array1->x * array1->y;
	nncuda_add_array_vector_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(dest->data, array1->data, array2->data, n, array1->x);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void nncuda_sub_array_kernel(float *dest, float *array1, float *array2, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		dest[tid] = array1[tid] - array2[tid];
}

void nncuda_sub_array(nncuda_array *dest, nncuda_array *array1, nncuda_array *array2) {
	if (!DIM_COMP(array1, array2)) {ERR("In add_array, shape of array1 and array2 are not equal\n");}
	if (!DIM_COMP(array2, dest)) {ERR("In add_array, shape of dest is not equal to array1 and array2\n");}
	uint32_t n = array1->x * array1->y;
	nncuda_sub_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(dest->data, array1->data, array2->data, n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void nncuda_add_array_kernel(float *dest, float *array1, float *array2, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		dest[tid] = array1[tid] + array2[tid];
}

void nncuda_add_array(nncuda_array *dest, nncuda_array *array1, nncuda_array *array2) {
	if (!DIM_COMP(array1, array2)) {ERR("In add_array, shape of array1 and array2 are not equal\n");}
	if (!DIM_COMP(array2, dest)) {ERR("In add_array, shape of dest is not equal to array1 and array2\n");}
	uint32_t n = array1->x * array1->y;
	nncuda_add_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(dest->data, array1->data, array2->data, n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void nncuda_dot_array_kernel(float *dest, float *A, float *B, uint32_t wA, uint32_t wB, uint32_t hA, uint32_t hB) {
	uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp = 0;
	if ((row < hA) && (col < wB)) {
		for (size_t i = 0, j = 0; i < wA, j < hB; i++, j++) {
			tmp += A[row * wA + i] * B[j * wB + col];
		}
		dest[row * wB + col] = tmp;
	}

}
void nncuda_dot_array(nncuda_array *dest, nncuda_array *array1, nncuda_array *array2) {
	if (array1->x != array2->y) {ERR("In dot_array, array1->x and array2->y are not equal\n");}
	if (array2->x != dest->x) {ERR("In dot_array, dest->x is not equal to array2->x\n");}
	if (array1->y != dest->y) {ERR("In dot_array, dest->y is not equal to array1->y\n");}
	dim3 threads (BLOCKSIZE, BLOCKSIZE);
	dim3 grid((array2->x + (BLOCKSIZE - 1)) / BLOCKSIZE, (array1->y + (BLOCKSIZE - 1))/ BLOCKSIZE);
	nncuda_dot_array_kernel<<<grid, threads>>>(dest->data, array1->data, array2->data, array1->x, array2->x, array1->y, array2->y);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void nncuda_ReLU_array_kernel(float *dest, float *A, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		dest[tid] = (A[tid] > 0.0) ? A[tid] : 0.0;
}

void nncuda_ReLU_array(nncuda_array *dest, nncuda_array *array) {
	if (!DIM_COMP(array, dest)) {ERR("In ReLU_array, shape of array and dest are not the same\n");}
	uint32_t n = array->x * array->y;
	nncuda_ReLU_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(dest->data, array->data, n);
}

__global__ void nncuda_softmax_array_kernel(float *dest, float *A, uint32_t wA, uint32_t hA) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < wA) {
		float sum = 0;
		float max = A[tid];
		for (size_t i = 0; i < hA; i++) {
			if (A[(i * wA) + tid] > max) max = A[(i * wA) + tid]; 
		}
		for (size_t i = 0; i < hA; i++) {
			sum += exp(A[(i * wA) + tid] - max);
		}
		for (size_t i = 0; i < hA; i++) {
			dest[(i * wA) + tid] = exp(A[(i * wA) + tid] - max) / sum;
		}
	}
}

void nncuda_softmax_array(nncuda_array *dest, nncuda_array *array) {
	if (!DIM_COMP(array, dest)) {ERR("In ReLU_array, shape of array and dest are not the same\n");}
	nncuda_softmax_array_kernel<<<GRIDSIZE(array->x), BLOCKSIZE>>>(dest->data, array->data, array->x, array->y);
}

nncuda_network *nncuda_init(nncuda_array *input, nncuda_array *labels) {
	nncuda_network *network = (nncuda_network*)calloc(1, sizeof(nncuda_network));
	network->input = nncuda_zero_array(input->x, input->y);
	gpuErrchk( cudaMalloc((void**)&network->input->data, network->input->size) );
	gpuErrchk( cudaMemcpy(network->input->data, input->data, input->size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaDeviceSynchronize() );
	network->labels = nncuda_zero_array(labels->x, labels->y);
	gpuErrchk( cudaMalloc((void**)&network->labels->data, network->labels->size) );
	gpuErrchk( cudaMemcpy(network->labels->data, labels->data, labels->size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaDeviceSynchronize() );
	srand(time(NULL));
	network->W[0] = nncuda_rand_array(784, 10);
	network->b[0] = nncuda_rand_array(1, 10);
	network->W[1] = nncuda_rand_array(10, 10);
	network->b[1] = nncuda_rand_array(1, 10);
	return network;
}

void nncuda_forward_prop(nncuda_network *network) {
	if (!network->Z[0]) network->Z[0] = nncuda_zero_array(network->input->x, network->W[0]->y);
	nncuda_dot_array(network->Z[0], network->W[0], network->input);
	nncuda_add_array_vector(network->Z[0], network->Z[0], network->b[0]);
	if (!network->A[0]) network->A[0] = nncuda_zero_array(network->Z[0]->x, network->Z[0]->y);
	nncuda_ReLU_array(network->A[0], network->Z[0]);
	if (!network->Z[1]) network->Z[1] = nncuda_zero_array(network->A[0]->x, network->W[1]->y);
	nncuda_dot_array(network->Z[1], network->W[1], network->A[0]);
	nncuda_add_array_vector(network->Z[1], network->Z[1], network->b[1]);
	if (!network->A[1]) network->A[1] = nncuda_zero_array(network->Z[1]->x, network->Z[1]->y);
	nncuda_softmax_array(network->A[1], network->Z[1]);
	/*
	nncuda_array *tmp1 = nncuda_rand_array(1, 4);
	nncuda_array *tmp2 = nncuda_rand_array(4, 4);
	nncuda_array *tmp3 = nncuda_rand_array(4, 4);
	nncuda_add_array_vector(tmp3, tmp2, tmp1);
	nncuda_print_array(tmp1);
	nncuda_print_array(tmp2);
	nncuda_print_array(tmp3);
	*/
}

void nncuda_back_prop(nncuda_network *network) {
}
