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
	void print() const;
	void T();
	void resize(std::pair<uint32_t, uint32_t> shape);
	void resize(uint32_t x, uint32_t y);
	std::pair<uint32_t, uint32_t> shape() const;
	uint32_t x() const;
	uint32_t y() const;
	void dot(Array A, Array B);
	void add(Array A);
	float *data() ;
	void copy(Array A);
	void ReLU(Array A);
	void softmax(Array A);
	~Array();
private:
	uint32_t _x, _y, _size;
	float *_data;
};

class Network {
public:
	Network(Array data, Array expected);
};
}


#endif //NNCUDA_CUH
