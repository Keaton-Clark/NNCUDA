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
#include <vector>
#include <iomanip>
#include <utility>

namespace nncuda 
{
class Array {
public:
	Array();
	Array(std::pair<uint32_t, uint32_t> shape, bool random);
	Array(uint32_t x, uint32_t y, bool random);
	Array(std::vector<std::vector<float>> matrix);
	Array(float *matrix, uint32_t x, uint32_t y);
	void print() const;
	void T();
	void resize(std::pair<uint32_t, uint32_t> shape);
	void resize(uint32_t x, uint32_t y);
	std::pair<uint32_t, uint32_t> shape() const;
	uint32_t x() const;
	uint32_t y() const;
	void dot(Array A, Array B);
	void add(Array A);
	void sub(Array A);
	void mult(float A);
	void mult(Array A);
	void sum(Array A);
	float *data();
	uint32_t size() const;
	void copy(Array A);
	void ReLU(Array A);
	void dReLU(Array A);
	void softmax(Array A);
	float accuracy();
private:
	uint32_t _x, _y, _size;
	float *_data;
};

class Network {
public:
	Network(Array input, Array expected, float alpha);
	void forward_prop();
	void back_prop();
	void update();
	float accuracy();
private:
	Array *W;
	Array *b;
	Array *Z;
	Array *A;
	Array *dZ;
	Array *dW;
	Array *db;
	Array _input;
	Array _expected;
	float _alpha;
	Array tmp;
};
}


#endif //NNCUDA_CUH
