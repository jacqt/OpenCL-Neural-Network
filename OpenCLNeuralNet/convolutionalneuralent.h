#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include "include.h"
#include "layer.h"

//Class describing the convolutional portion of a neural network
class ConvolutionalNetworkPortion
{
public:
    ConvolutionalLayer* convolutionalLayer;
    cl::Buffer cLayerBuffer;
    cl::Buffer outputsBuffer;
    cl::Buffer inputsBuffer;
    cl::Buffer convolveResultBuffer;
    //If the input is a CNN or another network, we will need the pointer to the 
    //  layers of the input network to calculate the output of the network
    cl::Buffer* outputLayersBuffer; //Pointer to the network that we connect to

    friend class NeuralNetwork;

    //Writes the network to a file
    void writeConvolutionalNetworkPortionToFile();

    //Loads a convolutional network from file
    void loadConvlutionalNeuralNetworkFromFile();

    //Creates the filters for the convolutional network
    void createConvolutionalNetwork(
        unsigned int newFilterDim, 
        unsigned int newFilterNumberSize,
        unsigned int newInputDim,
        unsigned int newInputVectorNumberSize,
        cl::Buffer* newOutputLayerBuffer);

    //Create memory buffers and kernels
    void createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program);

    //Set the output layer buffer
    void setOutputLayerBuffer(cl::Buffer* newOutputLayerBuffer);

    //Get size of net
    size_t getSizeOfNet();

    //Computes the output of the network applied to a two dimensional input vector
    void computeOutput(cl_float* inputs, cl::CommandQueue *queue);

    //Computes the output of the network from the outputs of an input neural network
    //NO NEED FOR THIS AT THE PRESENT MOMENT
  //void computeOutputWithInputNet(cl::Buffer* inputLayerBuffer);

    //Trains the network given a set of inputs
    void trainConvolutionalPortion(cl::CommandQueue *queue);

private:
    size_t sizeOfNet;
    size_t sizeOfInput;
    size_t sizeOfOutput;
    size_t sizeOfConvolveResult;
    int inputDim;
    int inputVectorNumberSize;
    int filterDim;
    int filterNumberSize;
    int convolveResultDim;
    int outputDim;
    cl::Kernel computeConvolveResult;
    cl::Kernel trainConvolutionalNetworkPortion;
    cl::Kernel poolConvolveResult;
};

//Class describing a series of convolutional layers combined together
class ConvolutionalNeuralNetwork
{
public:
    vector<ConvolutionalNetworkPortion> layers;

    void createConvolutionalNeuralNetwork(
		const vector<vector<int> > &newConvolutionalNetSpec,
		const cl::Program convolutionalProgram);

private:
    vector<vector<int> > convolutionalNetSpec;
};
#endif
