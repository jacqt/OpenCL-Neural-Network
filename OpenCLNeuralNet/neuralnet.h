#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <tuple>
#include "layer.h"
#include <string>

using std::vector;
using std::cout;
using std::cin;
using std::endl;
using std::string;

//Reads a file
std::string getFileContents(const char* fileName);


class NeuralNet
{
public:
    vector<cl_int> netSpec;
    vector<Layer> layers;
    cl::Buffer netSpecBuffer;
    cl::Buffer layersBuffer;
    cl::Buffer inputBuffer;
    cl::Buffer targetBuffer;

    //Create the neural net as a vector of layers
    void createNeuralNet(vector<cl_int> &netSpec);

    //Loads a net from a file
    void loadNeuralNetFromFile(std::string netFileName);

    //Writes a net to a local file
	void writeNeuralNetToFile(cl::CommandQueue &queue);

    //Creates the memory buffers for the neural net and kernels
	void createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program);

    //Returns how large the net is in temrs of bytes
    int getSizeOfNet();

    //Computese the output of the neural net
    void computeOutput(
        cl::Context *context,
        cl_float *inputs,
        cl::Program *program,
        cl::CommandQueue *queue);

    //Given test data, calculates the error rate of the neural net
    void calculateError(
        cl::Context *context,
        vector<std::tuple<float*,int*> > *trainingData,
        cl::Program *program,
        cl::CommandQueue *queue);

    //Trains the neural net given a vector of tuples containing the feature vector
    //  and target vector
    void trainNeuralNet(
        cl::Context *context,
        vector<std::tuple<float*, int*> > *trainingData,
        cl::Program *program,
        cl::CommandQueue *queue,
        int trainingIterations);

private:
    size_t sizeOfNet;
    size_t sizeOfInput;
    size_t sizeOfTarget;
    cl::Kernel setInputKernel;
    cl::Kernel computeOutputRolled;
    cl::Kernel computeOutputUnrolled;
    cl::Kernel calcLayerDeltasRolled;
    cl::Kernel calcLayerDeltasUnrolled;
    cl::Kernel calcOutputLayerDeltas;
};

//Gets some test data
vector<std::tuple<float*, int*> > getTestData ();