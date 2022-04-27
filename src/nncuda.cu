#include "../include/nncuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define DIM_COMP(_ARRAY1, _ARRAY2) ((_ARRAY1->x == _ARRAY2->x) && (_ARRAY1->y ==  _ARRAY2->y))
#define ERR(_STR) fprintf(stderr, _STR); exit(EXIT_FAILURE)

#define BLOCKSIZE 32
#define GRIDSIZE(_ITERATIONS) (_ITERATIONS + (BLOCKSIZE - 1)) / BLOCKSIZE

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
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

nncuda::Array::~Array() {gpuErrchk( cudaFree(_data) );}

nncuda::Array::Array() : _x(0), _y(0), _size(0) {}

nncuda::Array::Array(std::pair<uint32_t, uint32_t> shape, bool random) : _x(shape.first), _y(shape.second) {
	_size = _x * _y * sizeof(float);
	gpuErrchk( cudaMalloc(&_data, _size) );
	float buff[_x * _y];
	if (random) {
		for (int i = 0; i < _x * _y; i++)
			buff[i] = ((double)rand()/(double)(RAND_MAX)) - .5;
	} else {
		memset(buff, 0, _x * _y * sizeof(float));
	}
	gpuErrchk( cudaMemcpy(_data, buff, _size, cudaMemcpyHostToDevice) );
}	

nncuda::Array::Array(uint32_t x, uint32_t y, bool random) : _x(x), _y(y) {
	_size = _x * _y * sizeof(float);
	gpuErrchk( cudaMalloc(&_data, _size) );
	float buff[_x * _y];
	if (random) {
		for (int i = 0; i < _x * _y; i++)
			buff[i] = ((double)rand()/(double)(RAND_MAX)) - .5;
			//buff[i] = 2.0;(float)(rand()%5);
	} else {
		memset(buff, 0, _x * _y * sizeof(float));
	}
	gpuErrchk( cudaMemcpy(_data, buff, _size, cudaMemcpyHostToDevice) );
}

nncuda::Array::Array(std::vector<std::vector<float>> matrix) : _x(matrix.size()), _y(matrix[0].size()) {
	_size = _x * _y * sizeof(float);
	gpuErrchk( cudaMalloc(&_data, _size) );
	float buff[_x * _y];
	for (int i = 0; i < _x; i++) {
		for (int j = 0; j < _y; j++) {
			buff[(j * _x) + i] = matrix[i][j];
		}
	}
	gpuErrchk( cudaMemcpy(_data, buff, _size, cudaMemcpyHostToDevice) );
}

void nncuda::Array::print() const {
	float tmp[_x * _y];
	gpuErrchk( cudaMemcpy(tmp, _data, _size, cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaDeviceSynchronize() );
	std::cout << "----------------------\n";
	for (int j = 0; j < _y; j++) {
		std::cout << std::setw(7) << j << ": ";
		if (_x > 10) {
			for (int i = 0; i < 8; i++) {
				std::cout << "|" << std::setw(12) << tmp[(_x * j) + i];
			}
			std::cout << "| ......... |" << std::setw(12) << tmp[(_x * j) + _x - 1];
		} else {
			for (int i = 0; i < _x; i++) {
				std::cout << "|" << std::setw(12) << tmp[(_x * j) + i];
			}
		}			
		std::cout << "|" << std::endl;
	}
	std::cout << "         ";
	if (_x > 10) {
		for (int i = 0; i < 8; i++) {
			std::cout << "|" << std::setw(11) << i << ":";
		}
		std::cout << "| ......... |" << std::setw(11) << _x-1 << ":";
	} else {
		for (int i = 0; i < _x; i++) {
			std::cout << "|" << std::setw(11) << i << ":";
		}
	}
	std::cout << "|" << std::endl;
	std::cout << "----------------------\n";
}

void nncuda::Array::T() {
	uint32_t x0 = _x;
	uint32_t y0 = _y;
	_y = x0;
	_x = y0;
	float tmp1[_x * _y];
	float tmp2[_x * _y];
	gpuErrchk( cudaMemcpy(tmp1, _data, _size, cudaMemcpyDeviceToHost) );
	for (int i = 0; i < x0; i++) {
		for (int j = 0; j < y0; j++) {
			tmp2[(i * _x) + j] = tmp1[(j * x0) + i];
		}
	}
	gpuErrchk( cudaMemcpy(_data, tmp2, _size, cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void nncuda::Array::resize(uint32_t x, uint32_t y) {
	_x = x;
	_y = y;
	if (_size == x * y * sizeof(float)) return;
	_size = _x * _y * sizeof(float);
	gpuErrchk( cudaFree(_data) );
	gpuErrchk( cudaMalloc(&_data, _size) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void nncuda::Array::resize(std::pair<uint32_t, uint32_t> shape) {
	resize(shape.first, shape.second);
}

std::pair<uint32_t, uint32_t> nncuda::Array::shape() const { return std::pair<uint32_t, uint32_t>(_x, _y); }

uint32_t nncuda::Array::x() const { return _x; }

uint32_t nncuda::Array::y() const { return _y; }

float *nncuda::Array::data() { return _data; }

void nncuda::Array::dot(nncuda::Array A, nncuda::Array B) {
	if (A.x() != B.y()) {ERR("In dot, A.x and B.y are not equal\n");}	
	resize(B.x(), A.y());
	dim3 threads(BLOCKSIZE, BLOCKSIZE);
	dim3 grid((_x + (BLOCKSIZE - 1)) / BLOCKSIZE, (_y + (BLOCKSIZE - 1)) / BLOCKSIZE);
	nncuda_dot_array_kernel<<<grid, threads>>>(_data, A.data(), B.data(), A.x(), B.x(), A.y(), B.y());
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void nncuda_add_array_vector_kernel(float *A, float *B, uint32_t count, uint32_t wA) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count)
		A[tid] = A[tid] + B[tid / wA];
}

void nncuda::Array::add(nncuda::Array A) {
	if (A.x() == 1) {
		if (_y != A.y()) {ERR("In add, A.y and self.y are not equal\n");}			
		uint32_t n = _x * _y;
		nncuda_add_array_vector_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n, _x);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
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

void nncuda::Array::copy(nncuda::Array A) {
	resize(A.x(), A.y());
	uint32_t n = _x * _y;
	nncuda_copy_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
void nncuda::Array::ReLU(nncuda::Array A) {
	resize(A.shape());
	uint32_t n = _x * _y;
	nncuda_ReLU_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
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

void nncuda::Array::softmax(nncuda::Array A) {
	resize(A.shape());
	nncuda_softmax_array_kernel<<<GRIDSIZE(_x), BLOCKSIZE>>>(_data, A.data(), _x, _y);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
