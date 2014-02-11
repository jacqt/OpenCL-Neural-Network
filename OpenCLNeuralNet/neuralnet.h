#ifndef NEURALNET_H
#define NEURALNET_H
#include "include.h"
#include "fullyconnectedneuralnet.h"
#include "convolutionalneuralent.h"

//Reads a file
std::string getFileContents(const char* fileName);

//Gets some test data
vector<std::tuple<float*, int*> > getTestData ();

//Class combining the CNN and fully connected neural networks into one structure
class NeuralNetwork
{
public:
    ConvolutionalNeuralNetwork cNN;
    FullyConnectedNeuralNet fNN;

};

#endif