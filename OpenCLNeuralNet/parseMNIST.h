#ifndef PARSE_MNIST_H
#define PARSE_MNIST_H
#include "include.h"

void readMNIST(vector<float*> &inputs, vector<int*> &targets, int &inputSize);
void readMNISTTest(vector<float*> &inputs, vector<int*> &targets, int &inputSize);
#endif