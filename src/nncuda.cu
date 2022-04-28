#include "../include/nncuda.cuh"
#include "../include/kernels.cuh"

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


nncuda::Array::Array() : _x(0), _y(0), _size(0) {
	gpuErrchk( cudaMalloc(&_data, _size) );
}

nncuda::Array::Array(float *matrix, uint32_t x, uint32_t y) : _x(x), _y(y) {
	_size = _x * _y * sizeof(float);
	gpuErrchk( cudaMalloc(&_data, _size) );
	gpuErrchk( cudaMemcpy(_data, matrix, _size, cudaMemcpyHostToDevice) );
}

nncuda::Array::Array(std::pair<uint32_t, uint32_t> shape, bool random) : _x(shape.first), _y(shape.second) {
	_size = _x * _y * sizeof(float);
	gpuErrchk( cudaMalloc(&_data, _size) );
	float buff[_x * _y];
	if (random) {
		for (int i = 0; i < _x * _y; i++)
			buff[i] = ((float)rand()/(float)(RAND_MAX)) - .5;
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
			buff[i] = ((float)rand()/(float)(RAND_MAX)) - .5;
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
	float *tmp1 = new float[_x * _y];
	float *tmp2 = new float[_x * _y];
	gpuErrchk( cudaMemcpy(tmp1, _data, _size, cudaMemcpyDeviceToHost) );
	for (int i = 0; i < x0; i++) {
		for (int j = 0; j < y0; j++) {
			tmp2[(i * _x) + j] = tmp1[(j * x0) + i];
		}
	}
	gpuErrchk( cudaMemcpy(_data, tmp2, _size, cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	delete(tmp1);
	delete(tmp2);
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

uint32_t nncuda::Array::size() const { return _size; }

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

void nncuda::Array::add(nncuda::Array A) {
	if (A.x() == 1) {
		if (_y != A.y()) {ERR("In add, A.y and self.y are not equal\n");}			
		uint32_t n = _x * _y;
		nncuda_add_array_vector_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n, _x);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
}

void nncuda::Array::mult(float A) {
	uint32_t n = _x * _y;
	nncuda_mult_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A, n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void nncuda::Array::mult(nncuda::Array A) {
	uint32_t n = _x * _y;
	nncuda_mult_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void nncuda::Array::sub(nncuda::Array A) {
	if (A.x() == 1) {

	} else {
		if (_y != A.y()) {ERR("In add, A.y and self.y are not equal\n");}			
		uint32_t n = _x * _y;
		nncuda_sub_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
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

void nncuda::Array::dReLU(nncuda::Array A) {
	resize(A.shape());
	uint32_t n = _x * _y;
	nncuda_dReLU_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_data, A.data(), n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void nncuda::Array::softmax(nncuda::Array A) {
	resize(A.shape());
	nncuda_softmax_array_kernel<<<GRIDSIZE(_x), BLOCKSIZE>>>(_data, A.data(), _x, _y);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void nncuda::Array::sum(nncuda::Array A) {
	resize(1, A.y());
	nncuda_sum_array_kernel<<<GRIDSIZE(_y), BLOCKSIZE>>>(_data, A.data(), A.x(), A.y());
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

nncuda::Network::Network(nncuda::Array input, nncuda::Array expected, float alpha) : _alpha(alpha) {
	srand(time(NULL));
	_input = input;
	_expected = expected;
	W = new nncuda::Array[2];
	b = new nncuda::Array[2];
	Z = new nncuda::Array[2];
	A = new nncuda::Array[2];
	dZ = new nncuda::Array[2];
	dW = new nncuda::Array[2];
	db = new nncuda::Array[2];
	W[0] = nncuda::Array(784, 10, true);
	b[0] = nncuda::Array(1, 10, true);
	W[1] = nncuda::Array(10, 10, true);
	b[1] = nncuda::Array(1, 10, true);
}

void nncuda::Network::forward_prop() {
	Z[0].dot(W[0], _input);
	Z[0].add(b[0]);
	A[0].ReLU(Z[0]);
	Z[1].dot(W[1], A[0]);
	Z[1].add(b[1]);
	A[1].softmax(Z[1]);
	A[1].print();
}

void nncuda::Network::back_prop() {
	dZ[1].copy(A[1]);
	dZ[1].sub(_expected);
	tmp.copy(A[0]);
	tmp.T();
	dW[1].dot(dZ[1], tmp);
	dW[1].mult(1.0/_expected.y());
	db[1].sum(dZ[1]);
	db[1].mult(1.0/_expected.y());
	//**********
	tmp.copy(W[1]);
	tmp.T();
	dZ[0].dot(tmp, dZ[1]);
	tmp.dReLU(Z[0]);
	dZ[0].mult(tmp);
	//********
	tmp.copy(_expected);
	tmp.T();
	dW[0].dot(dZ[0], tmp);
	dW[0].mult(1.0/_expected.y());
	db[0].sum(dZ[0]);
	db[0].mult(1.0/_expected.y());
}

void nncuda::Network::update() {
	dW[0].mult(_alpha);
	W[0].sub(dW[0]);
	dW[1].mult(_alpha);
	W[1].sub(dW[1]);
	db[0].mult(_alpha);
	b[0].sub(db[0]);
	db[1].mult(_alpha);
	b[1].sub(db[1]);
}

float nncuda::Network::accuracy() {
	float h_sum = 0;
	float *d_sum;
	gpuErrchk( cudaMalloc(&d_sum, sizeof(float)) );
	gpuErrchk( cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice) );
	uint32_t n = _expected.x() * _expected.y();
	nncuda_accuracy_array_kernel<<<GRIDSIZE(n), BLOCKSIZE>>>(_expected.data(), A[1].data(), d_sum, n);
	gpuErrchk( cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost) );
	return h_sum/_expected.x();
}
