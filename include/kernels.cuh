#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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

__global__ void nncuda_mult_array_kernel(float *A, float B, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		A[tid] = A[tid] * B;
}

__global__ void nncuda_mult_array_kernel(float *A, float *B, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		A[tid] = A[tid] * B[tid];
}

__global__ void nncuda_sub_array_kernel(float *A, float *B, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		A[tid] = A[tid] - B[tid];
}

__global__ void nncuda_add_array_vector_kernel(float *A, float *B, uint32_t count, uint32_t wA) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		A[tid] = A[tid] + B[tid / wA];
}

__global__ void nncuda_copy_array_kernel(float *A, float *B, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		A[tid] = B[tid];
}

__global__ void nncuda_ReLU_array_kernel(float *dest, float *A, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		dest[tid] = (A[tid] > 0.0) ? A[tid] : 0.0;
}

__global__ void nncuda_accuracy_array_kernel(float *A, float *B, float *sum, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count) {
		if (A[tid]) {
			*sum += B[tid];
		}
	}
}

__global__ void nncuda_dReLU_array_kernel(float *dest, float *A, uint32_t count) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		dest[tid] = (A[tid] > 0.0) ? 1.0 : 0.0;
}

__global__ void nncuda_sum_array_kernel(float *A, float *B, uint32_t wB, uint32_t hB) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < hB)
		for (int i = 0; i < wB; i++)
			A[tid] += B[tid * wB + i];
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

