#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H
#include "include.h"
#include "layer.h"

//Class describing a fully connected neural net
class FullyConnectedNeuralNet
{
public:
    vector<cl_int> netSpec;
    vector<Layer> layers;
    cl::Buffer netSpecBuffer;
    cl::Buffer layersBuffer;
    cl::Buffer inputBuffer;
    cl::Buffer outputBuffer;
    cl::Buffer targetBuffer;
    //If the input is a CNN or another network, we will need the pointer to the 
    //  outputs of the input network to calculate the output of the network
    cl::Buffer *inputNetOutputBuffer;

    //Create the neural net as a vector of layers
    void createFullyConnectedNeuralNet(vector<cl_int> &netSpec);

    //Loads a net from a file
    void loadFullyConnectedNeuralNetFromFile(std::string netFileName);

    //Writes a net to a local file
    void writeFullyConnectedNeuralNetToFile(cl::CommandQueue &queue);

    //Creates the memory buffers for the neural net and kernels
    void createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program);

    //Returns how large the net is in temrs of bytes
    int getSizeOfNet();

    //Computese the output of the neural net given an array of inputs
    void computeOutput(
        cl::Context *context,
        cl_float *inputs,
        cl::Program *program,
        cl::CommandQueue *queue);

    //Computes the output of the neural net with the inputs being the
    //  outputs of another neural network
    void computeOutputWithInputNet(
        cl::Context *context,
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
    void trainFullyConnectedNeuralNet(
        cl::Context *context,
        vector<std::tuple<float*, int*> > *trainingData,
        cl::Program *program,
        cl::CommandQueue *queue,
        int trainingIterations);

private:
    int lastLayerIndex;
    size_t sizeOfNet;
    size_t sizeOfInput;
    size_t sizeOfTarget;
    size_t sizeOfOutput;
    cl::Kernel setInputKernel;
    cl::Kernel computeOutputRolled;
    cl::Kernel computeOutputUnrolled;
    cl::Kernel calcLayerErrorGradientsRolled;
    cl::Kernel calcLayerErrorGradientsUnrolled;
    cl::Kernel calcOutputLayerErrorGradients;
    cl::Kernel writeOutputToBuffer;
};
#endif