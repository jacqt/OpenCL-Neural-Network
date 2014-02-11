#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include "include.h"
#include "layer.h"

//Class describing a convultional neural net
class ConvolutionalNeuralNetwork
{
public:
    vector<cl_int> netSpec;
    ConvolutionalLayer convolutionalLayer;
    cl::Buffer layersBuffer;
    cl::Buffer outputsBuffer;
    cl::Buffer inputBuffer;
    //If the input is a CNN or another network, we will need the pointer to the 
    //  layers of the input network to calculate the output of the network
    cl::Buffer* inputNetOutputBuffer;

    //Writes the network to a file
    void writeConvolutionalNeuralNetworkToFile();

    //Loads a convolutional network from file
    void loadConvlutionalNeuralNetworkFromFile();

    //Creates the filters for the convolutional network
    void createConvolutionalNetwork(unsigned int newFilterDim, 
        unsigned int newFilterNumber,
        unsigned int newInputDim);

    //Create memory buffers and kernels
    void createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program);

    //Get size of net
    int getSizeOfNet();

    //Computes the output of the network applied to a two dimensional input vector
    void computeOutput(vector<vector<float> > &inputs);

    //Computes the output of the network from the outputs of an input neural network
    void computeOutputWithInputNet(vector<vector<float> > &inputs);

    //Uses the outputLayersBuffer to get the error gradients which it uses to train the network.
    //  Thus no arguments are required
    void trainNetwork();

private:
    size_t sizeOfNet;
    size_t sizeOfInput;
    int filterDim;
    int filterNumber;
    int inputDim;
    cl::Kernel convolveFilter;
};
#endif
