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

//Create the neural net as a vector of layers
void createNeuralNet( 
    vector<cl_int> &netSpec,
    vector<Layer> &layers);

//Loads a net from a file
void loadNeuralNetFromFile(
    std::string netFileName,
    vector<cl_int> &netSpec,
    vector<Layer> &layers);

//Returns how large the net is in temrs of bytes
int getSizeOfNet(vector <Layer> &layers);


//Computese the output of the neural net
void computeOutput(
    cl::Context *context,
    cl_float *inputs,
    vector<int> *netSpec,
    vector<Layer> *layers,
    cl::Program *program,
    cl::Buffer *netSpecBuffer,
    cl::Buffer *layersBuffer,
    cl::CommandQueue *queue);

//Given test data, calculates the error rate of the neural net
void calculateError(
    cl::Context *context,
    vector<std::tuple<float*,int*> > *trainingData,
    vector<int> *netSpec,
    vector<Layer> *layers,
    cl::Program *program,
    cl::Buffer *netSpecBuffer,
    cl::Buffer *layersBuffer,
    cl::CommandQueue *queue);

//Trains the neural net given a vector of tuples containing the feature vector
//  and target vector
void trainNeuralNet(
    cl::Context *context,
    vector<std::tuple<float*, int*> > *trainingData,
    vector<int> *netSpec,
    vector<Layer> *layers,
    cl::Program *program,
    cl::Buffer *netSpecBuffer,
    cl::Buffer *layersBuffer,
    cl::CommandQueue *queue,
    int trainingIterations);

//Gets some test data
vector<std::tuple<float*, int*> > getTestData ();