#ifndef PARSE_MNIST_H
#define PARSE_MNIST_H
#include "include.h"
#include "neuralnet.h"

void readMNIST(vector<float*> &inputs, vector<int*> &targets, int &inputSize);
void readMNISTTest(vector<float*> &inputs, vector<int*> &targets, int &inputSize);

void printInput(float* inputs);

//Functions to train and test a fully connected NN 
void trainMNISTFullyConnectedNN();
void testMNISTFullyConnectedNN();


//Functions to train and test a convolutioanl NN
void trainMNISTConvolutionalNN();
void testMNISTConvolutionalNN();

#endif